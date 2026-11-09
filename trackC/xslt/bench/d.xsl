<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:output method="text"/>
<xsl:template match="/d"><xsl:for-each select="m/r"><xsl:value-of select="sum(v)"/><xsl:text>&#10;</xsl:text></xsl:for-each></xsl:template>
</xsl:stylesheet>
