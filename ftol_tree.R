# Construct a dataset-labelled tree from FTOL, recording name resolution and grafts.
build_ftol_tree <- function(species) {
  for (pkg in c('ftolr', 'ape', 'phytools', 'pteridocat', 'taxastand'))
    if (!requireNamespace(pkg, quietly = TRUE)) stop(paste('Missing R package:', pkg))
  species <- unique(gsub(' ', '_', trimws(as.character(species))))
  if (anyNA(species) || any(!nzchar(species))) stop('Invalid species names')
  suppressPackageStartupMessages(library(ftolr))
  tree <- ftolr::ft_tree()
  if (!inherits(tree, 'phylo') || anyDuplicated(tree$tip.label) ||
      is.null(tree$edge.length) || !ape::is.ultrametric(tree, tol=1e-7)) stop('Invalid or non-ultrametric FTOL tree')
  original <- tree
  binomial <- function(x) vapply(strsplit(gsub('_', ' ', x), '\\s+'), function(z) paste(head(z, 2), collapse='_'), character(1))
  genus <- function(x) sub('_.*', '', x)
  ft_bins <- binomial(original$tip.label)
  ft_genera <- genus(original$tip.label)
  utils::data('pteridocat', package='pteridocat', envir=environment())
  resolved <- taxastand::ts_resolve_names(query=gsub('_', ' ', species), ref_taxonomy=taxastand::ts_format_ref(pteridocat))
  mapping <- data.frame(species=species, ftol_tip=NA_character_, placement=NA_character_, stringsAsFactors=FALSE)
  for (i in seq_along(species)) {
    matches <- which(ft_bins == binomial(species[i]))
    if (length(matches) == 1) {
      mapping$ftol_tip[i] <- original$tip.label[matches]; mapping$placement[i] <- 'exact binomial'
    } else {
      candidates <- resolved$resolved_name[resolved$query == gsub('_', ' ', species[i]) & !is.na(resolved$resolved_name)]
      matches <- which(ft_bins %in% binomial(candidates))
      if (length(matches) == 1) {
        mapping$ftol_tip[i] <- original$tip.label[matches]; mapping$placement[i] <- 'synonym'
      } else mapping$placement[i] <- if (length(matches)>1) 'ambiguous' else 'unmatched'
    }
  }
  # Shared FTOL tips cannot be relabelled to two dataset taxa without inventing relationships.
  shared <- duplicated(mapping$ftol_tip) | duplicated(mapping$ftol_tip, fromLast=TRUE)
  shared <- shared & !is.na(mapping$ftol_tip)
  mapping$placement[shared] <- 'shared FTOL tip unresolved'
  mapping$ftol_tip[shared] <- NA_character_
  grafted <- character()
  bind_at <- function(tree, sp, where, position=0) {
    heights <- ape::node.depth.edgelength(tree)
    tip_height <- max(heights[seq_along(tree$tip.label)])
    length <- tip_height - heights[where] + position
    phytools::bind.tip(tree, sp, where=where, position=position, edge.length=length)
  }
  for (i in which(is.na(mapping$ftol_tip) & !shared)) {
    sp <- species[i]; g <- genus(sp); gtips <- original$tip.label[ft_genera == g]
    if (!length(gtips)) { mapping$placement[i] <- 'genus absent'; next }
    if (length(gtips)>1) {
      mrca <- ape::getMRCA(original, gtips)
      if (!setequal(ape::extract.clade(original, mrca)$tip.label, gtips)) {
        mapping$placement[i] <- 'genus non-monophyletic'; next
      }
      tree <- bind_at(tree, sp, where=ape::getMRCA(tree, gtips))
      mapping$placement[i] <- 'grafted at genus MRCA'
    } else {
      previous <- grafted[genus(grafted)==g]
      if (length(previous)) {
        tree <- bind_at(tree, sp, where=ape::getMRCA(tree, c(gtips,previous)))
        mapping$placement[i] <- 'grafted at genus polytomy'
      } else {
        tip <- match(gtips, tree$tip.label); edge <- which(tree$edge[,2]==tip); len <- tree$edge.length[edge]
        if (length(len)!=1 || !is.finite(len) || len<=0) { mapping$placement[i] <- 'invalid terminal length'; next }
        tree <- bind_at(tree, sp, where=tip, position=len/2)
        mapping$placement[i] <- 'grafted at terminal branch'
      }
    }
    mapping$ftol_tip[i] <- sp; grafted <- c(grafted, sp)
  }
  keep <- !is.na(mapping$ftol_tip)
  tree <- ape::keep.tip(tree, mapping$ftol_tip[keep])
  tree$tip.label <- mapping$species[match(tree$tip.label, mapping$ftol_tip)]
  if (anyNA(tree$tip.label) || anyDuplicated(tree$tip.label)) stop('Invalid final tip labels')
  list(newick=ape::write.tree(tree), mapping=mapping, ftol_version=as.character(ftolr::ft_data_ver()))
}
