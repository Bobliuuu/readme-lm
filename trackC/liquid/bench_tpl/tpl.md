---
layout: null
permalink: /b.txt
---
{%- assign W = site.data.b.W -%}{%- assign x = site.data.b.x -%}
{%- for rep in (1..REPS) -%}
{%- for i in (0..127) -%}{%- assign acc = 0.0 -%}{%- assign row = W[i] -%}{%- for j in (0..127) -%}{%- assign acc = x[j] | times: row[j] | plus: acc -%}{%- endfor -%}{%- endfor -%}
{%- endfor -%}
{{ acc }}
