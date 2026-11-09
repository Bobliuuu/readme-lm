require_relative "shim"
require "github/markup"
puts GitHub::Markup.render("x.mediawiki", "{{#expr: 7+5}} / {{#ifexpr: 7+5=12|T|F}} / {{#expr: 1*5+2*6+3*7+4*8}}")
puts GitHub::Markup.render("x.adoc", "{counter2:s:7}{counter2:s}{counter2:s}{counter2:s}{counter2:s}{counter2:s}\n\nM1 s={s}\n\n:a: 7\n:b: 5\n\nifeval::[{a} > {b}]\nA-GT-B\nendif::[]\n\nifeval::[{a} + {b} == 12]\nARITH\nendif::[]\n\ninclude::c.txt[]\n")
