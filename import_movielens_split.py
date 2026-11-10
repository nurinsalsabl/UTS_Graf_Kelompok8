"""
MovieLens -> Neo4j dengan data dipisah 80% latih / 20% uji,
lalu membentuk komunitas FILM dengan Leiden dan Louvain (dari data latih saja).

Yang dibuat di Neo4j:
  (:User)-[:RATED {rating}]->(:Movie)        <- 80% rating (LATIH)
  (:User)-[:RATED_TEST {rating}]->(:Movie)   <- 20% rating (UJI, tidak dipakai membentuk komunitas)
  (:Movie)-[:HAS]->(:Genre)
  (:Movie)-[:SIMILAR_COS {weight}]->(:Movie) <- kemiripan cosine, dihitung dari data LATIH
  Movie.leidenId, Movie.louvainId            <- nomor komunitas

Aturan pemisahan SAMA dengan milik Nurma:
  rating masuk UJI jika (userId*7 + movieId*13) % 5 == 0

Contoh pakai:
  pip install -U neo4j
  python import_movielens_split.py -p PASSWORDMU -s data --clear
"""
import argparse
import csv
import os
import sys
import time

from neo4j import GraphDatabase

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--user", default="neo4j")
parser.add_argument("-p", "--password", required=True)
parser.add_argument("-b", "--uri", default="bolt://localhost:7687")
parser.add_argument("-s", "--source", default="data", help="folder berisi movies.csv & ratings.csv")
parser.add_argument("-d", "--database", default="neo4j")
parser.add_argument("--clear", action="store_true", help="hapus SEMUA data di database dulu")
parser.add_argument("--min-rating", type=float, default=4.0, help="rating dianggap 'suka'")
parser.add_argument("--min-cooccur", type=int, default=5, help="minimal user yang menyukai kedua film")
parser.add_argument("--min-cos", type=float, default=0.15, help="ambang cosine untuk membuat garis")
parser.add_argument("--repeat", type=int, default=5, help="jumlah pengulangan untuk rentang modularity")
parser.add_argument("--skip-gds", action="store_true", help="hanya import + split, tanpa Louvain/Leiden")
args = parser.parse_args()

movies_path = os.path.join(args.source, "D:\\graf\\import-movielens\\data\\movies.csv")
ratings_path = os.path.join(args.source, "D:\\graf\\import-movielens\\data\\ratings.csv")
for p in (movies_path, ratings_path):
    if not os.path.isfile(p):
        sys.exit(f"File tidak ditemukan: {p}")

BATCH = 2000


def chunks(rows, size=BATCH):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
try:
    driver.verify_connectivity()
except Exception as e:
    sys.exit(f"Gagal konek ke Neo4j: {e}\nCek: database sudah Start? password benar? port 7687?")
print("Terkoneksi ke Neo4j.")

start = time.time()
with driver.session(database=args.database) as s:

    def run(q, **p):
        return s.run(q, **p).consume()

    def one(q, **p):
        return s.run(q, **p).single()

    if args.clear:
        print("Menghapus data lama...")
        run("MATCH (n) DETACH DELETE n")
        try:
            run("CALL gds.graph.drop('cosGraph', false)")
        except Exception:
            pass

    for q in (
        "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE",
        "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.genre IS UNIQUE",
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE",
    ):
        run(q)

    # 1. Film + genre ---------------------------------------------------------
    print("Import film...")
    movies = []
    with open(movies_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            genres = [] if r["genres"] == "(no genres listed)" else r["genres"].split("|")
            movies.append({"movieId": int(r["movieId"]), "title": r["title"], "genres": genres})
    q_movies = """
        UNWIND $rows AS row
        MERGE (m:Movie {movieId: row.movieId})
        SET m.title = row.title
        WITH m, row
        UNWIND row.genres AS g
        MERGE (ge:Genre {genre: g})
        MERGE (m)-[:HAS]->(ge)
    """
    for part in chunks(movies):
        run(q_movies, rows=part)
    print(f"  {len(movies)} film")

    # 2. Rating, dipisah latih / uji -------------------------------------------
    print("Import rating (80% latih, 20% uji)...")
    train, test = [], []
    with open(ratings_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            row = {
                "userId": int(r["userId"]),
                "movieId": int(r["movieId"]),
                "rating": float(r["rating"]),
                "timestamp": int(r["timestamp"]),
            }
            if (row["userId"] * 7 + row["movieId"] * 13) % 5 == 0:
                test.append(row)
            else:
                train.append(row)

    q_train = """
        UNWIND $rows AS row
        MATCH (m:Movie {movieId: row.movieId})
        MERGE (u:User {userId: row.userId})
        MERGE (u)-[r:RATED]->(m)
        SET r.rating = row.rating, r.timestamp = row.timestamp
    """
    q_test = q_train.replace("[r:RATED]", "[r:RATED_TEST]")
    for part in chunks(train):
        run(q_train, rows=part)
    for part in chunks(test):
        run(q_test, rows=part)
    total = len(train) + len(test)
    print(f"  latih: {len(train)} ({100 * len(train) / total:.1f}%) | uji: {len(test)} ({100 * len(test) / total:.1f}%)")

    # 3. Graf kemiripan film (cosine) dari data LATIH saja ---------------------
    print("Membangun graf film (cosine) dari data latih...")
    run("MATCH ()-[c:SIMILAR_COS]->() DELETE c")
    run("MATCH (m:Movie) REMOVE m.likes")
    run("""
        MATCH (m:Movie)<-[r:RATED]-(:User) WHERE r.rating >= $minr
        WITH m, count(*) AS n
        SET m.likes = n
    """, minr=args.min_rating)
    run("""
        MATCH (m1:Movie)<-[r1:RATED]-(u:User)-[r2:RATED]->(m2:Movie)
        WHERE m1.movieId < m2.movieId AND r1.rating >= $minr AND r2.rating >= $minr
        WITH m1, m2, count(u) AS w
        WHERE w >= $minco
        WITH m1, m2, w / sqrt(toFloat(m1.likes) * m2.likes) AS cos
        WHERE cos >= $mincos
        MERGE (m1)-[c:SIMILAR_COS]->(m2)
        SET c.weight = cos
    """, minr=args.min_rating, minco=args.min_cooccur, mincos=args.min_cos)

    run("MATCH (m:Terhubung) REMOVE m:Terhubung")
    run("MATCH (m:Movie) WHERE (m)-[:SIMILAR_COS]-() SET m:Terhubung")
    n_total = one("MATCH (m:Movie) RETURN count(m) AS n")["n"]
    n_conn = one("MATCH (m:Terhubung) RETURN count(m) AS n")["n"]
    n_edge = one("MATCH ()-[c:SIMILAR_COS]->() RETURN count(c) AS n")["n"]
    print(f"  film terhubung: {n_conn} dari {n_total} ({100 * n_conn / n_total:.1f}%) | garis: {n_edge}")

    if args.skip_gds:
        print("--skip-gds aktif, selesai sampai di sini.")
        driver.close()
        sys.exit(0)

    # 4. Leiden dan Louvain (terpisah, di graf yang SAMA) -----------------------
    run("CALL gds.graph.drop('cosGraph', false)")
    run("""CALL gds.graph.project('cosGraph', 'Terhubung',
           {SIMILAR_COS: {orientation: 'UNDIRECTED', properties: 'weight'}})""")

    hasil = {}
    r = one("""CALL gds.leiden.write('cosGraph',
               {relationshipWeightProperty: 'weight', writeProperty: 'leidenId',
                randomSeed: 42, concurrency: 1})
               YIELD communityCount, modularity, ranLevels""")
    hasil["leidenId"] = ("Leiden", r["communityCount"], r["modularity"], r["ranLevels"])
    r = one("""CALL gds.louvain.write('cosGraph',
               {relationshipWeightProperty: 'weight', writeProperty: 'louvainId'})
               YIELD communityCount, modularity, ranLevels""")
    hasil["louvainId"] = ("Louvain", r["communityCount"], r["modularity"], r["ranLevels"])

    # Rentang modularity dari beberapa kali jalan
    rentang = {"Leiden": [], "Louvain": []}
    for i in range(1, args.repeat + 1):
        r = one("""CALL gds.leiden.stats('cosGraph',
                   {relationshipWeightProperty: 'weight', randomSeed: $seed, concurrency: 1})
                   YIELD modularity""", seed=i)
        rentang["Leiden"].append(r["modularity"])
        r = one("""CALL gds.louvain.stats('cosGraph', {relationshipWeightProperty: 'weight'})
                   YIELD modularity""")
        rentang["Louvain"].append(r["modularity"])

    # 5. Ringkasan -------------------------------------------------------------
    print("\n===== HASIL (graf film dari data latih) =====")
    print(f"Film terhubung: {n_conn} dari {n_total} | garis: {n_edge}")
    for prop, (nama, cc, q, lv) in hasil.items():
        print(f"\n{nama}: communityCount={cc}  modularity={q:.4f}  ranLevels={lv}")
        vals = rentang[nama]
        print(f"  rentang modularity ({args.repeat}x jalan): {min(vals):.4f} - {max(vals):.4f}")
        sizes = [rec for rec in s.run(
            f"MATCH (m:Terhubung) RETURN m.{prop} AS c, count(*) AS n ORDER BY n DESC")]
        nyata = [x for x in sizes if x["n"] > 1]
        print(f"  komunitas berisi >1 film: {len(nyata)}")
        print("  5 komunitas terbesar:")
        for x in nyata[:5]:
            c, n = x["c"], x["n"]
            gen = [g["genre"] for g in s.run(
                f"MATCH (m:Movie) WHERE m.{prop} = $c MATCH (m)-[:HAS]->(g:Genre) "
                "RETURN g.genre AS genre, count(*) AS n ORDER BY n DESC LIMIT 3", c=c)]
            fil = [t["title"] for t in s.run(
                f"MATCH (m:Movie) WHERE m.{prop} = $c RETURN m.title AS title "
                "ORDER BY m.likes DESC LIMIT 3", c=c)]
            print(f"    #{c}: {n} film | genre: {', '.join(gen)} | contoh: {' ; '.join(fil)}")

driver.close()
print(f"\nSelesai dalam {time.time() - start:.1f} detik.")
