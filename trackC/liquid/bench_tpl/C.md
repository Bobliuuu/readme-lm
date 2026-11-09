{%- assign W = site.data.b.W4 -%}{%- assign x = site.data.b.x4 -%}
{%- for rep in (1..20) -%}
{%- for row in W -%}{%- assign acc = 0.0 -%}{%- for w in row -%}{%- assign xg = x[forloop.index0] -%}{%- assign acc = xg[0] | times: w[0] | plus: acc | plus: xg[1] | times: w[1] -%}{%- endfor -%}{%- endfor -%}
{%- endfor -%}
{{ acc }}
