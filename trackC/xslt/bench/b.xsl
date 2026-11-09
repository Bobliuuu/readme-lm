<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:exsl="http://exslt.org/common">
<xsl:output method="text"/>
<xsl:template match="/d">
  <xsl:variable name="x" select="x/v"/>
  <xsl:for-each select="m/r">
    <xsl:variable name="rr" select="v"/>
    <xsl:variable name="p"><xsl:for-each select="$x"><p><xsl:value-of select=". * $rr[position()=1]"/></p></xsl:for-each></xsl:variable>
    <xsl:value-of select="count(v)"/><xsl:text>&#10;</xsl:text>
  </xsl:for-each>
</xsl:template>
</xsl:stylesheet>
