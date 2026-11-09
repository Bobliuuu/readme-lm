<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:exsl="http://exslt.org/common" xmlns:math="http://exslt.org/math">
<xsl:import href="../lib.xsl"/>
<xsl:output method="text"/>
<xsl:template match="/">
vendor=<xsl:value-of select="system-property('xsl:vendor')"/> version=<xsl:value-of select="system-property('xsl:version')"/>
exsl:node-set available=<xsl:value-of select="function-available('exsl:node-set')"/>
math:exp available=<xsl:value-of select="function-available('math:exp')"/>
exsl:node-set test=<xsl:value-of select="count(exsl:node-set(/t/*))"/>
<xsl:variable name="e"><xsl:call-template name="exp"><xsl:with-param name="x" select="-1.5"/></xsl:call-template></xsl:variable>
own exp(-1.5)=<xsl:value-of select="$e"/> (true 0.22313016014842982)
</xsl:template>
</xsl:stylesheet>
