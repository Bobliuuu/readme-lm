{%- assign W = site.data.b.W -%}{%- assign x = site.data.b.x -%}
{%- for rep in (1..20) -%}
{%- for row in W -%}{%- assign acc = 0.0 -%}{%- for w in row -%}{%- assign acc = x[forloop.index0] | times: w | plus: acc -%}{%- endfor -%}{%- endfor -%}
{%- endfor -%}
{{ acc }}
