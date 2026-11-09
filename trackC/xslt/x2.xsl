<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:exsl="http://exslt.org/common" xmlns:math="http://exslt.org/math" exclude-result-prefixes="exsl math">
<xsl:import href="lib.xsl"/>
<xsl:output method="text"/>
<xsl:template match="/t">
 <xsl:variable name="x" select="x/v"/><xsl:variable name="g" select="g/v"/>
 <xsl:text>relu </xsl:text>
 <xsl:variable name="r"><xsl:call-template name="relu-vec"><xsl:with-param name="x" select="$x"/></xsl:call-template></xsl:variable>
 <xsl:for-each select="exsl:node-set($r)/v"><xsl:value-of select="."/><xsl:text> </xsl:text></xsl:for-each><xsl:text>&#10;gelu </xsl:text>
 <xsl:variable name="ge"><xsl:call-template name="gelu-vec"><xsl:with-param name="x" select="$x"/></xsl:call-template></xsl:variable>
 <xsl:for-each select="exsl:node-set($ge)/v"><xsl:value-of select="."/><xsl:text> </xsl:text></xsl:for-each><xsl:text>&#10;softmax </xsl:text>
 <xsl:variable name="sm"><xsl:call-template name="softmax"><xsl:with-param name="x" select="$x"/></xsl:call-template></xsl:variable>
 <xsl:for-each select="exsl:node-set($sm)/v"><xsl:value-of select="."/><xsl:text> </xsl:text></xsl:for-each><xsl:text>&#10;exp </xsl:text>
 <xsl:for-each select="$x"><xsl:call-template name="exp"><xsl:with-param name="x" select="."/></xsl:call-template><xsl:text> </xsl:text></xsl:for-each><xsl:text>&#10;mathexp </xsl:text>
 <xsl:for-each select="$x"><xsl:value-of select="math:exp(.)"/><xsl:text> </xsl:text></xsl:for-each><xsl:text>&#10;rms </xsl:text>
 <xsl:variable name="rn"><xsl:call-template name="rms"><xsl:with-param name="x" select="x2/v"/><xsl:with-param name="g" select="x2g/v"/></xsl:call-template></xsl:variable>
 <xsl:for-each select="exsl:node-set($rn)/v"><xsl:value-of select="."/><xsl:text> </xsl:text></xsl:for-each><xsl:text>&#10;sqrt </xsl:text>
 <xsl:for-each select="sq/v"><xsl:call-template name="sqrt"><xsl:with-param name="a" select="."/></xsl:call-template><xsl:text> </xsl:text></xsl:for-each>
</xsl:template>
</xsl:stylesheet>
