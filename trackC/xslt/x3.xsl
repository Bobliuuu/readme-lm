<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:exsl="http://exslt.org/common" exclude-result-prefixes="exsl">
<xsl:import href="lib.xsl"/>
<xsl:output method="text"/>
<xsl:variable name="m" select="/m"/>
<xsl:template match="/m">
  <xsl:call-template name="gen"><xsl:with-param name="tok" select="@start"/><xsl:with-param name="s" select="s0/v"/><xsl:with-param name="n" select="@steps"/></xsl:call-template>
</xsl:template>

<!-- one step: returns <t>token</t><v>new state</v>... -->
<xsl:template name="step"><xsl:param name="tok"/><xsl:param name="s"/>
  <xsl:variable name="e" select="$m/E/r[$tok+1]/c/v"/>
  <xsl:variable name="h0"><xsl:call-template name="add"><xsl:with-param name="a" select="$e"/><xsl:with-param name="b" select="$s"/></xsl:call-template></xsl:variable>
  <xsl:variable name="n1"><xsl:call-template name="rms"><xsl:with-param name="x" select="exsl:node-set($h0)/v"/><xsl:with-param name="g" select="$m/g/v"/></xsl:call-template></xsl:variable>
  <xsl:variable name="a1"><xsl:call-template name="matvec"><xsl:with-param name="M" select="$m/W1"/><xsl:with-param name="x" select="exsl:node-set($n1)/v"/><xsl:with-param name="n" select="128"/></xsl:call-template></xsl:variable>
  <xsl:variable name="r1"><xsl:call-template name="relu-vec"><xsl:with-param name="x" select="exsl:node-set($a1)/v"/></xsl:call-template></xsl:variable>
  <xsl:variable name="h1"><xsl:call-template name="add"><xsl:with-param name="a" select="exsl:node-set($h0)/v"/><xsl:with-param name="b" select="exsl:node-set($r1)/v"/></xsl:call-template></xsl:variable>
  <xsl:variable name="n2"><xsl:call-template name="rms"><xsl:with-param name="x" select="exsl:node-set($h1)/v"/><xsl:with-param name="g" select="$m/g/v"/></xsl:call-template></xsl:variable>
  <xsl:variable name="a2"><xsl:call-template name="matvec"><xsl:with-param name="M" select="$m/W2"/><xsl:with-param name="x" select="exsl:node-set($n2)/v"/><xsl:with-param name="n" select="128"/></xsl:call-template></xsl:variable>
  <xsl:variable name="r2"><xsl:call-template name="relu-vec"><xsl:with-param name="x" select="exsl:node-set($a2)/v"/></xsl:call-template></xsl:variable>
  <xsl:variable name="h2"><xsl:call-template name="add"><xsl:with-param name="a" select="exsl:node-set($h1)/v"/><xsl:with-param name="b" select="exsl:node-set($r2)/v"/></xsl:call-template></xsl:variable>
  <xsl:variable name="ones"><xsl:for-each select="$m/g/v"><v>1</v></xsl:for-each></xsl:variable>
  <xsl:variable name="nf"><xsl:call-template name="rms"><xsl:with-param name="x" select="exsl:node-set($h2)/v"/><xsl:with-param name="g" select="exsl:node-set($ones)/v"/></xsl:call-template></xsl:variable>
  <xsl:variable name="lg"><xsl:call-template name="matvec"><xsl:with-param name="M" select="$m/H"/><xsl:with-param name="x" select="exsl:node-set($nf)/v"/><xsl:with-param name="n" select="128"/></xsl:call-template></xsl:variable>
  <!-- argmax -->
  <xsl:variable name="lgs" select="exsl:node-set($lg)/v"/>
  <xsl:variable name="best"><xsl:for-each select="$lgs"><xsl:sort select="." data-type="number" order="descending"/><xsl:if test="position()=1"><xsl:value-of select="count(preceding-sibling::v)"/></xsl:if></xsl:for-each></xsl:variable>
  <t><xsl:value-of select="$best"/></t>
  <xsl:for-each select="exsl:node-set($h2)/v"><xsl:variable name="i" select="position()"/><v><xsl:value-of select="0.5*$s[$i] + 0.05*."/></v></xsl:for-each>
</xsl:template>

<xsl:template name="gen"><xsl:param name="tok"/><xsl:param name="s"/><xsl:param name="n"/>
  <xsl:value-of select="$tok"/><xsl:text> </xsl:text>
  <xsl:if test="$n &gt; 0">
    <xsl:variable name="r"><xsl:call-template name="step"><xsl:with-param name="tok" select="$tok"/><xsl:with-param name="s" select="$s"/></xsl:call-template></xsl:variable>
    <xsl:variable name="rn" select="exsl:node-set($r)"/>
    <xsl:call-template name="gen"><xsl:with-param name="tok" select="$rn/t"/><xsl:with-param name="s" select="$rn/v"/><xsl:with-param name="n" select="$n - 1"/></xsl:call-template>
  </xsl:if>
</xsl:template>
</xsl:stylesheet>
