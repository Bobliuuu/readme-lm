{%- assign W = site.data.b.W4 -%}{%- assign x = site.data.b.x4 -%}
{%- for rep in (1..20) -%}
{%- for row in W -%}{%- assign acc = 0.0 -%}{%- for w in row -%}{%- assign xg = x[forloop.index0] -%}{%- assign acc = xg[0] | times: w[0] | plus: acc -%}{%- assign acc = xg[1] | times: w[1] | plus: acc -%}{%- assign acc = xg[2] | times: w[2] | plus: acc -%}{%- assign acc = xg[3] | times: w[3] | plus: acc -%}{%- endfor -%}{%- endfor -%}
{%- endfor -%}
{{ acc }}
