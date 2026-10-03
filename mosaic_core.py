"""Headless structural projection core. No Tk or matplotlib imports."""
from __future__ import annotations
import ast,json,re
from dataclasses import asdict,dataclass,replace
from hashlib import sha256
from typing import Mapping,Any
SHA=re.compile(r"^[0-9a-f]{64}$")
class GraphProjectionError(ValueError):pass
@dataclass(frozen=True)
class GraphProjection:
 schema:str;source_ref:str;source_digest:str;nodes:tuple[tuple[str,str],...];edges:tuple[tuple[str,str,str],...];input_class:str;upstream_digest:str|None;authority_effect:str="NONE";projection_digest:str=""
 def payload(self):d=asdict(self);d.pop("projection_digest",None);return d
 def compute(self):return sha256(b"LION/MOSAIC-GRAPH/1\0"+json.dumps(self.payload(),sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
 def validate(self,req=True):
  if self.schema!="lion.graph-projection/v1" or not self.source_ref:raise GraphProjectionError("identity")
  if SHA.fullmatch(self.source_digest) is None:raise GraphProjectionError("source_digest")
  if self.input_class not in {"PYTHON_SOURCE","GLITCHLAB_DELTA"}:raise GraphProjectionError("input_class")
  if self.upstream_digest is not None and SHA.fullmatch(self.upstream_digest) is None:raise GraphProjectionError("upstream_digest")
  if tuple(sorted(set(self.nodes)))!=self.nodes or tuple(sorted(set(self.edges)))!=self.edges:raise GraphProjectionError("canonical graph")
  if self.authority_effect!="NONE":raise GraphProjectionError("authority")
  if req and (SHA.fullmatch(self.projection_digest) is None or self.projection_digest!=self.compute()):raise GraphProjectionError("digest")
  return self
 def sealed(self):return replace(self,projection_digest=self.compute()).validate()
class V(ast.NodeVisitor):
 def __init__(self):self.nodes=set();self.edges=set();self.stack=[]
 def visit_FunctionDef(self,n):
  q=".".join(self.stack+[n.name]);self.nodes.add((q,"function"))
  if self.stack:self.edges.add((".".join(self.stack),q,"contains"))
  self.stack.append(n.name);self.generic_visit(n);self.stack.pop()
 visit_AsyncFunctionDef=visit_FunctionDef
 def visit_Call(self,n):
  if self.stack:
   if isinstance(n.func,ast.Name):name=n.func.id
   elif isinstance(n.func,ast.Attribute):name=n.func.attr
   else:name="<dynamic>"
   dst="call:"+name;self.nodes.add((dst,"call"));self.edges.add((".".join(self.stack),dst,"calls"))
  self.generic_visit(n)
def project_python_source(source:str,*,source_ref:str)->GraphProjection:
 if not isinstance(source,str):raise GraphProjectionError("source")
 raw=source.encode();tree=ast.parse(source);v=V();v.visit(tree)
 return GraphProjection("lion.graph-projection/v1",source_ref,sha256(raw).hexdigest(),tuple(sorted(v.nodes)),tuple(sorted(v.edges)),"PYTHON_SOURCE",None).sealed()
def project_glitchlab_delta(value:Mapping[str,Any])->GraphProjection:
 required={"schema","repository_ref","base_commit","head_commit","changed_files","diff_sha256","delta_histogram","delta_fingerprint","invariant_score","invariant_block","provider_source_ref","process_semantics_ref","process_semantics_digest","authority_effect","mutation_effect","observation_digest"}
 if not isinstance(value,Mapping) or set(value)!=required or value.get("schema")!="lion.glitchlab-delta-observation/v1":raise GraphProjectionError("delta shape")
 up=value.get("observation_digest")
 if not isinstance(up,str) or SHA.fullmatch(up) is None:raise GraphProjectionError("delta digest")
 nodes={(f"file:{p}","file") for p in value["changed_files"]}
 nodes|={(f"token:{k}","delta_token") for k,_ in value["delta_histogram"]}
 edges=set()
 for p in value["changed_files"]:
  for k,_ in value["delta_histogram"]:edges.add((f"file:{p}",f"token:{k}","observed_delta"))
 raw=json.dumps(dict(value),sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
 return GraphProjection("lion.graph-projection/v1","glitchlab:"+value["repository_ref"],sha256(raw).hexdigest(),tuple(sorted(nodes)),tuple(sorted(edges)),"GLITCHLAB_DELTA",up).sealed()
