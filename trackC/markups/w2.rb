require_relative "shim"
require "wikicloth"
def r(s, &blk)
  c = Class.new(WikiCloth::Parser); c.class_eval(&blk) if blk
  c.new(data: s).to_html(noedit: true).strip
rescue => e
  "ERR #{e.class}: #{e.message[0,80]}"
end
T = {
 "nested"=>"{{#expr: {{#expr: 2*3}} + 1}}",
 "ops"=>"{{#expr: 7 mod 5}}|{{#expr: 2^10}}|{{#expr: 7/2}}|{{#expr: -3*4}}|{{#expr: (1+2)*(3+4)}}|{{#expr: 5>3}}|{{#expr: 1e3}}|{{#expr: round(2.5)}}|{{#expr: abs(-3)}}|{{#expr: max(1,2)}}|{{#expr: sqrt(16)}}",
 "relu_pos"=>"{{#ifexpr: -3 > 0 | -3 | 0}} {{#ifexpr: 5 > 0 | 5 | 0}}",
 "relu_nested"=>"{{#ifexpr: {{#expr: 1*2-3*4}} > 0 | {{#expr: 1*2-3*4}} | 0}}",
 "dot4"=>"{{#expr: 1*5 + 2*6 + 3*7 + 4*8}}",
 "matvec" => "{{#expr: 1*1+2*2+3*3+4*4}},{{#expr: 0*1+1*2+0*3+1*4}},{{#expr: 2*1-1*2+0*3+3*4}},{{#expr: 1*1+1*2+1*3+1*4}}",
 "switch_lut"=>"{{#switch: {{#expr: 2+3}} | 4=four | 5=five | #default=other}}",
 "ifeq_num"=>"{{#ifeq: {{#expr: 7+5}} | 12 | EQ | NE}}",
 "ifeq_num2"=>"{{#ifeq: {{#expr: 7+5}} | 12.0 | EQ | NE}}",
 "param_outside"=>"{{{1|dflt}}}",
 "tmpl_inline"=>"<template 1=\"X\">{{{1}}}</template>",
 "noinclude"=>"<includeonly>{{#expr:1+1}}</includeonly><noinclude>NI</noinclude>",
 "lua"=>"{{#luaexpr: 1+1}}",
 "vardefine"=>"{{#vardefine:a|3}}{{#var:a}}",
}
T.each{|k,v| puts "#{k}: #{v}\n   => #{r(v).inspect}"}
puts "--- emulated include handler (NOT available on GitHub; github-markup passes no handler)"
tm = {"Add"=>"{{#expr: {{{1}}} + {{{2}}} }}",
      "Dec"=>"{{#ifexpr: {{{1}}} <= 0 | 0 | {{Dec|{{#expr: {{{1}}}-1}}}} }}",
      "Sum"=>"{{#ifexpr: {{{1}}} <= 0 | 0 | {{#expr: {{{1}}} + {{Sum|{{#expr: {{{1}}}-1}} }} }} }}",
      "Self"=>"{{Self}}"}
h = Class.new(WikiCloth::WikiLinkHandler){ define_method(:include_resource){|r,o| tm[r] || ""} }
[ "{{Add|7|5}}", "{{Sum|3}}", "{{Sum|10}}", "{{Self}}"].each{|s|
  out = begin
    WikiCloth::Parser.new(data: s, link_handler: h.new).to_html(noedit: true).strip
  rescue => e; "ERR #{e.class} #{e.message[0,80]}"; end
  puts "#{s} => #{out[0,300].inspect}"}
