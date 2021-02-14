use super::{DecisionTree, TreeNode};
use linfa::{Float, Label};
use std::fmt::Debug;

/// Struct to print a fitted decision tree in LaTex using tikz and forest.
///
/// ### Usage
///
/// ```rust
/// use linfa::prelude::*;
/// use linfa_datasets;
/// use linfa_trees::DecisionTree;
///
/// // Load dataset
/// let dataset = linfa_datasets::iris();
/// // Fit the tree
/// let tree = DecisionTree::params().fit(&dataset);
/// // Export to tikz
/// let tikz = tree.export_to_tikz();
/// let latex_tree = tikz.to_string();
/// // Now you can write latex_tree to the preferred destination
///
/// ```
pub struct Tikz<'a, F: Float, L: Label + Debug> {
    legend: bool,
    max_classes: usize,
    complete: bool,
    tree: &'a DecisionTree<F, L>,
}

impl<'a, F: Float, L: Debug + Label> Tikz<'a, F, L> {
    pub fn new(tree: &'a DecisionTree<F, L>) -> Self {
        Tikz {
            legend: true,
            max_classes: 4,
            complete: true,
            tree,
        }
    }

    fn format_node(&self, node: &'a TreeNode<F, L>) -> String {
        let depth = vec![""; node.depth() + 1].join("\t");
        if let Some(prediction) = node.prediction() {
            format!("{}[Label: {:?}]", depth, prediction)
        } else {
            let (idx, value, impurity_decrease) = node.split();
            let mut out = format!(
                "{}[Val(${}$) $ \\geq {:.2}$ \\\\ Imp. ${:.2}$",
                depth, idx, value, impurity_decrease
            );
            for child in node.children().into_iter().filter_map(|x| x.as_ref()) {
                out.push('\n');
                out.push_str(&self.format_node(child));
            }
            out.push(']');

            out
        }
    }

    /// Whether a complete Tex document should be generated
    pub fn complete(mut self, complete: bool) -> Self {
        self.complete = complete;

        self
    }

    /// Add a legend to the generated tree
    pub fn with_legend(mut self) -> Self {
        self.legend = true;

        self
    }

    /// The maximal number of classes printed in each node
    pub fn max_classes(mut self, max_classes: usize) -> Self {
        self.max_classes = max_classes;

        self
    }
}

use std::fmt;

impl<'a, F: Float, L: Debug + Label> fmt::Display for Tikz<'a, F, L> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut out = String::from(
            r#"
\documentclass[margin=10pt]{standalone}
\usepackage{tikz,forest}
\usetikzlibrary{arrows.meta}

\forestset{
default preamble={
before typesetting nodes={
  !r.replace by={[, coordinate, append]}
},  
where n children=0{
  tier=word,
}{  
  %diamond, aspect=2,
},  
where level=0{}{
  if n=1{
    edge label={node[pos=.2, above] {Y}},
  }{  
    edge label={node[pos=.2, above] {N}},
  }   
},  
for tree={
  edge+={thick, -Latex},
  s sep'+=2cm,
  draw,
  thick,
  edge path'={ (!u) -| (.parent)},
  align=center,
}   
}
}

\begin{document}
\begin{forest}"#,
        );

        out.push_str(&self.format_node(self.tree.root_node()));
        out.push_str("\n\t\\end{forest}\n\\end{document}");

        write!(f, "{}", out)
    }
}
