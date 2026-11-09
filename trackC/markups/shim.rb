require "twitter-text"
module Twitter; Autolink = TwitterText::Autolink unless defined?(Twitter::Autolink); end rescue nil
class File; class << self; alias_method :exists?, :exist? unless respond_to?(:exists?); end; end
