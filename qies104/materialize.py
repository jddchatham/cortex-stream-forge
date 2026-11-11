"""Create admission-only source overlays. No numerical operations changed."""
from pathlib import Path
import json
R=Path(__file__).resolve().parent;W=R.parent
catalog=json.loads((R/'fixture-catalog.json').read_text())
(R/'fixture-catalog.inc').write_text(''.join('{'+json.dumps(p)+','+json.dumps(h)+'},\n' for p,h in sorted(catalog.items())))
s=(W/'qies099/ricursive-layer.cpp').read_text();s='#include "admission.h"\n'+s
s=s.replace('ModelFile::ModelFile(const std::string & path) {','ModelFile::ModelFile(const std::string & path) {\n    q104::real("ModelFile constructor");')
needle='    validate_specs(c, specs); validate_kv(c, previous);'
assert s.count(needle)==1
s=s.replace(needle,'    if (!c.fixture || source.is_model()) q104::real("direct r99::execute");\n    q104::synthetic();\n    require(c.hidden<=256 && c.ffn<=512 && c.heads<=4 && c.kv_heads<=2 && c.head<=64, \"QIES104 synthetic geometry budget\");\n'+needle)
(R/'ricursive-layer.cpp').write_text(s);(R/'ricursive-layer.h').write_text((W/'qies099/ricursive-layer.h').read_text())
s=(W/'qies102/bundle.cpp').read_text();s='#include "admission.h"\n'+s
s=s.replace('if(real)admission();','if(real)q104::real("LayerSource constructor");else for(auto&e:b.entries)q104::fixture_file(e.path);')
start=s.index('void admission(){');end=s.index('\n}\n}',start)+2;s=s[:start]+s[end:]
(R/'bundle.cpp').write_text(s);(R/'bundle.h').write_text((W/'qies102/bundle.h').read_text())
s=(W/'qies102/probe.cpp').read_text();s='#include "admission.h"\n'+s
needle='int main(int argc,char**argv)try{';assert needle in s
s=s.replace(needle,needle+'\n if(argc<2)q104::real("missing dispatch");\n if(std::string(argv[1])=="--synthetic")q104::synthetic();\n else if(std::string(argv[1])!="--describe")q104::real("candidate CLI dispatch");')
(R/'candidate.cpp').write_text(s)
s=(W/'qies100/candidate_probe.cpp').read_text();s='#include "admission.h"\n'+s
s=s.replace('if(argc!=3)return 2;', 'q104::synthetic();if(argc!=3)return 2;')
s=s.replace('source.root=argv[1];','source.root=argv[1];q104::fixture_directory(source.root);')
(R/'synthetic_candidate.cpp').write_text(s)
s=(W/'qies103/stock_oneblock_probe.cpp').read_text();s='#include "admission.h"\n'+s
s=s.replace('int main(int argc, char ** argv)', 'int main(int argc, char ** argv)')
needle='int main(int argc,char**argv){';assert needle in s
s=s.replace(needle,'int main(int argc,char**argv)try{\n if(argc>1&&std::string(argv[1])=="--real")q104::real("stock reference CLI");\n q104::synthetic();')
s=s.replace('std::string file=argv[1];','std::string file=argv[1];q104::fixture_file(file);')
s=s.replace('if(std::getenv("QIES100_REFERENCE_PLAIN"))mp.use_extra_bufts=false;','mp.use_extra_bufts=false;')
s+='\ncatch(const std::exception&e){std::cerr<<e.what()<<"\\n";return 1;}\n'
(R/'reference.cpp').write_text(s)
print('Materialized admission-only overlays; original numerical source preserved')
