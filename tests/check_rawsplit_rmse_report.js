// Exercise report rendering/filter handlers without a browser service.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const file = process.argv[2];
const html = fs.readFileSync(file, 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const values = {region:'all',case:'all',target:'all',horizon:'all',seed:'mean'};
const elems = {};
for (const id of ['region','case','target','horizon','seed','head','body','count']) {
  elems[id] = {value:values[id],innerHTML:'',textContent:'',options:[],handlers:{},
    add(option){this.options.push(option)},
    addEventListener(type,fn){this.handlers[type]=fn}};
}
class Option { constructor(text,value) { this.text=text; this.value=value; } }
vm.runInNewContext(script,{document:{getElementById:id=>elems[id]},Option}, {timeout:5000});
assert.match(elems.count.textContent,/456/);
assert.equal((elems.body.innerHTML.match(/<tr>/g)||[]).length,456);
assert.equal(elems.horizon.options.length,19);
elems.region.value='internal';elems.region.handlers.change();
assert.match(elems.count.textContent,/152/);
elems.case.value='R5';elems.case.handlers.change();
assert.match(elems.count.textContent,/38/);
elems.target.value='B35_H2S';elems.target.handlers.change();
assert.match(elems.count.textContent,/19/);
elems.horizon.value='H18';elems.horizon.handlers.change();
assert.match(elems.count.textContent,/1/);
assert.equal((elems.body.innerHTML.match(/<strong/g)||[]).length,5);
elems.seed.value='42';elems.seed.handlers.change();
assert.match(elems.body.innerHTML, /0\.00%/);
assert.ok(!elems.body.innerHTML.includes('±'));
console.log('PASS: 456 rows, all filters and seed 42 No-Gain baseline');
