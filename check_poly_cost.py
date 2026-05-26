import json, re
with open('src/testcases/case_ieee39.rs', 'r', encoding='utf-8') as f:
    content = f.read()
start = content.find('r#"') + 3
end = content.rfind('"#')
j = json.loads(content[start:end])
obj = j['_object']
if 'poly_cost' in obj:
    pc = obj['poly_cost']
    print(json.dumps(pc, indent=2)[:3000])
else:
    print('No poly_cost. Keys:', list(obj.keys()))
