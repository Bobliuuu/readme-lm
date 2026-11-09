require_relative "shim"
require "wikicloth"
puts WikiCloth::VERSION rescue nil
tests = {
 "expr" => "{{#expr: 7+5}}",
 "if" => "{{#if: x | YES | NO}}",
 "ifeq" => "{{#ifeq: 12 | 12 | EQ | NE}}",
 "switch" => "{{#switch: b | a=A | b=B | #default=D}}",
 "ifexpr" => "{{#ifexpr: 7+5 = 12 | T | F}}",
 "template_undefined" => "{{Foo|1}}",
 "magic_lc" => "{{lc:ABC}} {{uc:abc}} {{padleft:7|3|0}}",
 "var" => "{{#vardefine:a|3}}{{#var:a}}",
 "include" => "{{:Main Page}}",
 "math" => "<math>1+1</math>",
 "pre" => "<nowiki>{{#expr:1}}</nowiki>",
 "tag" => "{{#tag:b|hi}}",
 "html" => "<script>alert(1)</script> <b>ok</b> <span style='x'>s</span>",
 "var2" => "{{#expr: (7*3)-2 }}",
}
tests.each{|k,v| 
  begin
    r = WikiCloth::Parser.new(data: v).to_html(noedit: true)
  rescue => e
    r = "ERR #{e.class}: #{e.message[0,100]}"
  end
  puts "#{k}: #{v.inspect} => #{r.strip.inspect}"
}
# templates via custom subclass
class W < WikiCloth::Parser
  template { |t| {"Add"=>"{{#expr: {{{1}}} + {{{2}}} }}", "Rec"=>"{{#if:{{{1}}}|x{{Rec|}}|end}}","Loop"=>"{{Loop}}"}[t] }
end
puts W.new(data: "{{Add|7|5}}").to_html(noedit: true)
