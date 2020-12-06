use super::{DecisionTree, TreeNode};
use linfa::{Float, Label};
use std::fmt::Debug;

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

    pub fn format_node(&self, node: &'a TreeNode<F, L>) -> String {
        let depth = vec![""; node.depth() + 1].join("\t");
        if let Some(prediction) = node.prediction() {
            format!("{}[Label: {:?}]", depth, prediction)
        } else {
            let (idx, value, impurity_decrease) = node.split();
            let mut out = format!("{}[Val(${}$) $ \\geq {:.2}$ \\\\ Imp. ${:.2}$", depth, idx, value, impurity_decrease);
            for child in node.childs().into_iter().filter_map(|x| x.as_ref()) {
                out.push_str("\n");
                out.push_str(&self.format_node(child));
            }
            out.push_str("]");

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

    pub fn to_string(self) -> String {
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

        out
    }
}
