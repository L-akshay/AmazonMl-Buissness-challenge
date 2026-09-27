import random
import string
import unittest
import duckdb
from src.normalize import conservative
from src.run_pipeline import normalization_sql


class NativeNormalizationTests(unittest.TestCase):
    def test_native_path_matches_python_and_unicode_keeps_its_semantics(self):
        rng=random.Random(42)
        values=[None,'','  A&B Corp.\t42/3 Main St  ','Straße','École SARL',
                'ＡＢＣ １２','कृष्णा ट्रेडर्स','İstanbul','e\u0301 & ﬁ', '한글 상점']
        values += [''.join(rng.choices(string.printable,k=rng.randrange(100))) for _ in range(200)]
        calls=[]
        def observed(value):
            calls.append(value)
            return conservative(value)
        db=duckdb.connect(config={'threads':1,'memory_limit':'128MB'})
        db.create_function('conservative_text',observed,['VARCHAR'],'VARCHAR')
        db.execute('CREATE TABLE examples(value VARCHAR)')
        db.executemany('INSERT INTO examples VALUES (?)',[(v,) for v in values])
        actual=[r[0] for r in db.execute('SELECT '+normalization_sql('value')+' FROM examples').fetchall()]
        self.assertEqual(actual,[conservative(v) for v in values])
        self.assertTrue(calls)
        self.assertTrue(all(not value.isascii() for value in calls))
        db.close()
