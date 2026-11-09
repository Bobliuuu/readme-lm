# probes max recursive-include depth using the same liquid-4.0.4 Jekyll uses
require 'jekyll'
[30,60,90,95,99,100,101].each do |n|
  src = <<~L
  {%- include rec.html n=#{n} -%}{{ rec_out }}
  L
  Dir.chdir(File.join(__dir__, "site"))
  s = Jekyll::Site.new(Jekyll.configuration("quiet"=>true, "safe"=>true))
  begin
    t = Liquid::Template.parse(src)
    info = {registers: {site: s, page: {}}}
    ctx = Liquid::Context.new({}, {}, {site: s, page: {}}) rescue nil
    out = t.render!(s.site_payload, registers: {site: s, page: {}})
    puts "n=#{n} -> #{out.strip}"
  rescue => e
    puts "n=#{n} -> #{e.class}: #{e.message[0,60]}"
  end
end
