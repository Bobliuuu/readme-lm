<?xml version="1.0"?>
<!-- QA transformer (3 layers, d=128, 4 heads, RoPE, RMSNorm, GELU, tied head) in pure XSLT 1.0 (+exsl:node-set). No JavaScript. -->
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:exsl="http://exslt.org/common" exclude-result-prefixes="exsl">
<xsl:import href="lib.xsl"/>
<xsl:output method="text" encoding="UTF-8"/>

<xsl:variable name="M" select="document('model.xml')/model"/>
<xsl:variable name="voc" select="string($M/voc)"/>
<xsl:variable name="eos" select="string-length($voc)"/>
<xsl:variable name="UP" select="'ABCDEFGHIJKLMNOPQRSTUVWXYZ'"/>
<xsl:variable name="LO" select="'abcdefghijklmnopqrstuvwxyz'"/>
<xsl:variable name="q0" select="translate(string(/ask/q), $UP, $LO)"/>
<xsl:variable name="q1" select="normalize-space(translate($q0, translate($q0, $voc, ''), ''))"/>
<xsl:variable name="q" select="substring($q1, 1, 140)"/>
<xsl:variable name="seq" select="concat('q: ', $q, '&#10;a: ', string(/ask/@forced))"/>
<xsl:variable name="nseq" select="string-length($seq)"/>
<xsl:variable name="maxn" select="number(/ask/@max)"/>
<xsl:variable name="dump" select="/ask/@dump = '1'"/>

<xsl:template match="/ask">
  <xsl:variable name="c0" select="substring($seq,1,1)"/>
  <xsl:call-template name="gen">
    <xsl:with-param name="tok" select="string-length(substring-before($voc,$c0))"/>
    <xsl:with-param name="pos" select="0"/>
    <xsl:with-param name="cache" select="$M/nothing"/>
  </xsl:call-template>
</xsl:template>

<xsl:template name="gen"><xsl:param name="tok"/><xsl:param name="pos"/><xsl:param name="cache"/><xsl:param name="n" select="0"/>
  <xsl:variable name="last" select="$pos + 1 &gt;= $nseq"/>
  <xsl:variable name="r"><xsl:call-template name="step"><xsl:with-param name="tok" select="$tok"/><xsl:with-param name="pos" select="$pos"/><xsl:with-param name="cache" select="$cache"/><xsl:with-param name="head" select="$last"/></xsl:call-template></xsl:variable>
  <xsl:variable name="rn" select="exsl:node-set($r)"/>
  <xsl:choose>
    <xsl:when test="not($last)">
      <xsl:call-template name="gen"><xsl:with-param name="tok" select="string-length(substring-before($voc, substring($seq,$pos+2,1)))"/><xsl:with-param name="pos" select="$pos+1"/><xsl:with-param name="cache" select="$cache | $rn/kr"/><xsl:with-param name="n" select="0"/></xsl:call-template>
    </xsl:when>
    <xsl:otherwise>
      <xsl:variable name="t" select="number($rn/t)"/>
      <xsl:if test="$dump and $n = 0"><xsl:text>LOGITS</xsl:text><xsl:for-each select="$rn/l"><xsl:text> </xsl:text><xsl:value-of select="."/></xsl:for-each><xsl:text>&#10;</xsl:text></xsl:if>
      <xsl:if test="$n = 0"><xsl:text>ARGMAX </xsl:text><xsl:value-of select="$t"/><xsl:text>&#10;ANSWER </xsl:text></xsl:if>
      <xsl:if test="$t != $eos and $n &lt; $maxn">
        <xsl:value-of select="substring($voc, $t + 1, 1)"/>
        <xsl:call-template name="gen"><xsl:with-param name="tok" select="$t"/><xsl:with-param name="pos" select="$pos+1"/><xsl:with-param name="cache" select="$cache | $rn/kr"/><xsl:with-param name="n" select="$n+1"/></xsl:call-template>
      </xsl:if>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<!-- one token through 3 blocks (+ head if requested); returns new KV rows <kr> and, if head, <t> argmax (and <l> logits if dump) -->
<xsl:template name="step"><xsl:param name="tok"/><xsl:param name="pos"/><xsl:param name="cache"/><xsl:param name="head"/>
  <xsl:variable name="x0" select="$M/emb/r[$tok+1]/c/v"/>
  <xsl:variable name="cs" select="$M/rope/p[$pos+1]/v"/>
  <xsl:variable name="r0"><xsl:call-template name="layer"><xsl:with-param name="x" select="$x0"/><xsl:with-param name="l" select="0"/><xsl:with-param name="cs" select="$cs"/><xsl:with-param name="cache" select="$cache"/></xsl:call-template></xsl:variable>
  <xsl:variable name="n0" select="exsl:node-set($r0)"/>
  <xsl:variable name="r1"><xsl:call-template name="layer"><xsl:with-param name="x" select="$n0/v"/><xsl:with-param name="l" select="1"/><xsl:with-param name="cs" select="$cs"/><xsl:with-param name="cache" select="$cache"/></xsl:call-template></xsl:variable>
  <xsl:variable name="n1" select="exsl:node-set($r1)"/>
  <xsl:variable name="r2"><xsl:call-template name="layer"><xsl:with-param name="x" select="$n1/v"/><xsl:with-param name="l" select="2"/><xsl:with-param name="cs" select="$cs"/><xsl:with-param name="cache" select="$cache"/></xsl:call-template></xsl:variable>
  <xsl:variable name="n2" select="exsl:node-set($r2)"/>
  <xsl:copy-of select="$n0/kr | $n1/kr | $n2/kr"/>
  <xsl:if test="$head">
    <xsl:variable name="nf"><xsl:call-template name="rms"><xsl:with-param name="x" select="$n2/v"/><xsl:with-param name="g" select="$M/gf/v"/></xsl:call-template></xsl:variable>
    <xsl:variable name="lg"><xsl:call-template name="matvec"><xsl:with-param name="M" select="$M/emb"/><xsl:with-param name="x" select="exsl:node-set($nf)/v"/><xsl:with-param name="n" select="128"/></xsl:call-template></xsl:variable>
    <xsl:variable name="lgs" select="exsl:node-set($lg)/v"/>
    <t><xsl:for-each select="$lgs"><xsl:sort select="." data-type="number" order="descending"/><xsl:if test="position()=1"><xsl:value-of select="count(preceding-sibling::v)"/></xsl:if></xsl:for-each></t>
    <xsl:if test="$dump"><xsl:for-each select="$lgs"><l><xsl:value-of select="."/></l></xsl:for-each></xsl:if>
  </xsl:if>
</xsl:template>

<xsl:template name="mkrow"><xsl:param name="l"/><xsl:param name="h"/><xsl:param name="k"/><xsl:param name="v"/>
  <kr l="{$l}" h="{$h}"><k><xsl:copy-of select="$k[position() &gt; 32*$h and position() &lt;= 32*$h+32]"/></k><vv><xsl:copy-of select="$v[position() &gt; 32*$h and position() &lt;= 32*$h+32]"/></vv></kr>
</xsl:template>

<xsl:template name="layer"><xsl:param name="x"/><xsl:param name="l"/><xsl:param name="cs"/><xsl:param name="cache"/>
  <xsl:variable name="L" select="$M/layer[$l+1]"/>
  <xsl:variable name="n1"><xsl:call-template name="rms"><xsl:with-param name="x" select="$x"/><xsl:with-param name="g" select="$L/g1/v"/></xsl:call-template></xsl:variable>
  <xsl:variable name="qkv"><xsl:call-template name="matvec"><xsl:with-param name="M" select="$L/qkv"/><xsl:with-param name="x" select="exsl:node-set($n1)/v"/><xsl:with-param name="n" select="128"/></xsl:call-template></xsl:variable>
  <xsl:variable name="qv" select="exsl:node-set($qkv)/v"/>
  <xsl:variable name="qr"><xsl:call-template name="rope"><xsl:with-param name="x" select="$qv[position() &lt;= 128]"/><xsl:with-param name="cs" select="$cs"/></xsl:call-template></xsl:variable>
  <xsl:variable name="qs"><xsl:for-each select="exsl:node-set($qr)/v"><v><xsl:value-of select=". * 0.17677669529663688"/></v></xsl:for-each></xsl:variable>
  <xsl:variable name="qsn" select="exsl:node-set($qs)/v"/>
  <xsl:variable name="kr"><xsl:call-template name="rope"><xsl:with-param name="x" select="$qv[position() &gt; 128 and position() &lt;= 256]"/><xsl:with-param name="cs" select="$cs"/></xsl:call-template></xsl:variable>
  <xsl:variable name="krn" select="exsl:node-set($kr)/v"/>
  <xsl:variable name="vn" select="$qv[position() &gt; 256]"/>
  <xsl:variable name="rows">
    <xsl:call-template name="mkrow"><xsl:with-param name="l" select="$l"/><xsl:with-param name="h" select="0"/><xsl:with-param name="k" select="$krn"/><xsl:with-param name="v" select="$vn"/></xsl:call-template>
    <xsl:call-template name="mkrow"><xsl:with-param name="l" select="$l"/><xsl:with-param name="h" select="1"/><xsl:with-param name="k" select="$krn"/><xsl:with-param name="v" select="$vn"/></xsl:call-template>
    <xsl:call-template name="mkrow"><xsl:with-param name="l" select="$l"/><xsl:with-param name="h" select="2"/><xsl:with-param name="k" select="$krn"/><xsl:with-param name="v" select="$vn"/></xsl:call-template>
    <xsl:call-template name="mkrow"><xsl:with-param name="l" select="$l"/><xsl:with-param name="h" select="3"/><xsl:with-param name="k" select="$krn"/><xsl:with-param name="v" select="$vn"/></xsl:call-template>
  </xsl:variable>
  <xsl:variable name="rowsn" select="exsl:node-set($rows)/kr"/>
  <xsl:variable name="all" select="$cache[@l=$l] | $rowsn"/>
  <xsl:variable name="ao">
    <xsl:call-template name="attn-head"><xsl:with-param name="q" select="$qsn[position() &lt;= 32]"/><xsl:with-param name="rows" select="$all[@h=0]"/></xsl:call-template>
    <xsl:call-template name="attn-head"><xsl:with-param name="q" select="$qsn[position() &gt; 32 and position() &lt;= 64]"/><xsl:with-param name="rows" select="$all[@h=1]"/></xsl:call-template>
    <xsl:call-template name="attn-head"><xsl:with-param name="q" select="$qsn[position() &gt; 64 and position() &lt;= 96]"/><xsl:with-param name="rows" select="$all[@h=2]"/></xsl:call-template>
    <xsl:call-template name="attn-head"><xsl:with-param name="q" select="$qsn[position() &gt; 96]"/><xsl:with-param name="rows" select="$all[@h=3]"/></xsl:call-template>
  </xsl:variable>
  <xsl:variable name="po"><xsl:call-template name="matvec"><xsl:with-param name="M" select="$L/o"/><xsl:with-param name="x" select="exsl:node-set($ao)/v"/><xsl:with-param name="n" select="128"/></xsl:call-template></xsl:variable>
  <xsl:variable name="xa"><xsl:call-template name="add"><xsl:with-param name="a" select="$x"/><xsl:with-param name="b" select="exsl:node-set($po)/v"/></xsl:call-template></xsl:variable>
  <xsl:variable name="xan" select="exsl:node-set($xa)/v"/>
  <xsl:variable name="n2"><xsl:call-template name="rms"><xsl:with-param name="x" select="$xan"/><xsl:with-param name="g" select="$L/g2/v"/></xsl:call-template></xsl:variable>
  <xsl:variable name="u"><xsl:call-template name="matvec"><xsl:with-param name="M" select="$L/u"/><xsl:with-param name="x" select="exsl:node-set($n2)/v"/><xsl:with-param name="n" select="128"/></xsl:call-template></xsl:variable>
  <xsl:variable name="g"><xsl:call-template name="gelu-vec"><xsl:with-param name="x" select="exsl:node-set($u)/v"/></xsl:call-template></xsl:variable>
  <xsl:variable name="dn"><xsl:call-template name="matvec"><xsl:with-param name="M" select="$L/dn"/><xsl:with-param name="x" select="exsl:node-set($g)/v"/><xsl:with-param name="n" select="512"/></xsl:call-template></xsl:variable>
  <xsl:copy-of select="$rowsn"/>
  <xsl:call-template name="add"><xsl:with-param name="a" select="$xan"/><xsl:with-param name="b" select="exsl:node-set($dn)/v"/></xsl:call-template>
</xsl:template>

<xsl:template name="attn-head"><xsl:param name="q"/><xsl:param name="rows"/>
  <xsl:variable name="sc"><xsl:for-each select="$rows"><s><xsl:call-template name="dot32"><xsl:with-param name="w" select="k/v"/><xsl:with-param name="x" select="$q"/></xsl:call-template></s></xsl:for-each></xsl:variable>
  <xsl:variable name="m"><xsl:for-each select="exsl:node-set($sc)/s"><xsl:sort select="." data-type="number" order="descending"/><xsl:if test="position()=1"><xsl:value-of select="."/></xsl:if></xsl:for-each></xsl:variable>
  <xsl:variable name="pr">
    <xsl:for-each select="$rows">
      <xsl:variable name="s"><xsl:call-template name="dot32"><xsl:with-param name="w" select="k/v"/><xsl:with-param name="x" select="$q"/></xsl:call-template></xsl:variable>
      <xsl:variable name="e"><xsl:call-template name="exp"><xsl:with-param name="x" select="$s - $m"/></xsl:call-template></xsl:variable>
      <xsl:call-template name="wv"><xsl:with-param name="e" select="$e"/><xsl:with-param name="vv" select="vv/v"/></xsl:call-template>
    </xsl:for-each>
  </xsl:variable>
  <xsl:call-template name="wsum"><xsl:with-param name="p" select="exsl:node-set($pr)"/></xsl:call-template>
</xsl:template>
</xsl:stylesheet>
