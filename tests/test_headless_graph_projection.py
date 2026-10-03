import importlib,sys,unittest
import mosaic_core
class GraphProjectionTests(unittest.TestCase):
 def test_import_is_headless(self):
  self.assertNotIn("tkinter",sys.modules);self.assertNotIn("matplotlib",sys.modules)
 def test_python_projection_is_deterministic(self):
  s="def a():\n    return b()\ndef b():\n    return 1\n";a=mosaic_core.project_python_source(s,source_ref="fixture.py");b=mosaic_core.project_python_source(s,source_ref="fixture.py")
  self.assertEqual(a,b);self.assertEqual(a.authority_effect,"NONE")
 def test_glitchlab_mapping_requires_exact_schema(self):
  v={"schema":"lion.glitchlab-delta-observation/v1","repository_ref":"fixture/repo","base_commit":"1"*40,"head_commit":"2"*40,"changed_files":["a.py"],"diff_sha256":"3"*64,"delta_histogram":[["ADD_FN",1]],"delta_fingerprint":"abcd","invariant_score":0.1,"invariant_block":False,"provider_source_ref":"glx","process_semantics_ref":"hmk9d","process_semantics_digest":"4"*64,"authority_effect":"NONE","mutation_effect":"NONE","observation_digest":"5"*64}
  p=mosaic_core.project_glitchlab_delta(v);self.assertEqual(p.input_class,"GLITCHLAB_DELTA");self.assertEqual(p.upstream_digest,"5"*64)
  v["extra"]=1
  with self.assertRaises(mosaic_core.GraphProjectionError):mosaic_core.project_glitchlab_delta(v)
if __name__=="__main__":unittest.main()
