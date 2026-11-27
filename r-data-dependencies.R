# R packages required by loader.py to reconstruct the fern phylogeny.
repos <- c(
  joelnitta = "https://joelnitta.r-universe.dev",
  CRAN = "https://cloud.r-project.org"
)

install.packages(c("ape", "phytools", "ftolr", "taxastand"), repos = repos)

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", repos = repos)
}
remotes::install_github("fernphy/pteridocat")
