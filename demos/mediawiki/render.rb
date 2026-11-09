# usage: ruby render.rb page.mediawiki out.html [github]  (shims from trackC/markups reused)
require File.expand_path("../../../trackC/markups/shim", __FILE__)
data = File.read(ARGV[0])
t = Process.clock_gettime(Process::CLOCK_MONOTONIC)
if ARGV[2] == "github"
  require "github/markup"
  html = GitHub::Markup.render(File.basename(ARGV[0]), data)
else
  require "wikicloth"
  html = WikiCloth::WikiCloth.new(data: data).to_html(noedit: true)
end
File.write(ARGV[1], html)
STDERR.puts "render_seconds=%.2f html_bytes=%d" % [Process.clock_gettime(Process::CLOCK_MONOTONIC) - t, html.bytesize]
