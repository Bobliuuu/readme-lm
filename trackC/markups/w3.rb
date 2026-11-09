require_relative "shim"
require "wikicloth"
tm = {"A"=>"{{#ifexpr: {{{1}}} <= 0 | END | a{{B|{{#expr: {{{1}}}-1}}}} }}",
      "B"=>"{{#ifexpr: {{{1}}} <= 0 | END | b{{A|{{#expr: {{{1}}}-1}}}} }}"}
h = Class.new(WikiCloth::WikiLinkHandler){ define_method(:include_resource){|r,o| tm[r] || ""} }
["{{A|1}}","{{A|4}}","{{A|40}}"].each{|s|
 out = begin; WikiCloth::Parser.new(data: s, link_handler: h.new).to_html(noedit: true).strip; rescue SystemStackError=>e; "STACK"; rescue=>e; "ERR #{e.message[0,80]}"; end
 puts "#{s} => #{out[0,200].inspect}"}
