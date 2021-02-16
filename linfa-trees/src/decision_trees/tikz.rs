use super::{DecisionTree, TreeNode};
use linfa::{Float, Label};
use std::collections::HashSet;
use std::fmt::Debug;

/// Struct to print a fitted decision tree in Tex using tikz and forest.
///
/// There are two settable parameters:
///
/// * `legend`: if true, a box with the names of the split features will appear in the top right of the tree
/// * `complete`: if true, a complete and standalone Tex document will be generated; otherwise the result will an embeddable
///  Tex tree.
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
/// let tikz = tree.export_to_tikz().with_legend();
/// let latex_tree = tikz.to_string();
/// // Now you can write latex_tree to the preferred destination
///
/// ```
pub struct Tikz<'a, F: Float, L: Label + Debug> {
    legend: bool,
    complete: bool,
    tree: &'a DecisionTree<F, L>,
}

impl<'a, F: Float, L: Debug + Label> Tikz<'a, F, L> {
    /// Creates a new Tikz structure for the decision tree
    /// with the following default parameters:
    ///
    /// * `legend=false`
    /// * `complete=true`
    pub fn new(tree: &'a DecisionTree<F, L>) -> Self {
        Tikz {
            legend: false,
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
                "{}[Val(${}$) $ \\leq {:.2}$ \\\\ Imp. ${:.2}$",
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

    fn legend(&self) -> String {
        if self.legend {
            let mut map = HashSet::new();
            let mut out = "\n".to_string()
                + r#"\node [anchor=north west] at (current bounding box.north east) {%
                \begin{tabular}{c c c}
                  \multicolumn{3}{@{}l@{}}{Legend:}\\
                  Imp.&:&Impurity decrease\\"#;
            for node in self.tree.iter_nodes() {
                if !node.is_leaf() && !map.contains(&node.split().0) {
                    let var = format!(
                        "Var({})&:&{}\\\\",
                        node.split().0,
                        node.feature_name().unwrap()
                    );
                    out.push_str(&var);
                    map.insert(node.split().0);
                }
            }
            out.push_str("\\end{tabular}};");
            out
        } else {
            "".to_string()
        }
    }
}

use std::fmt;

impl<'a, F: Float, L: Debug + Label> fmt::Display for Tikz<'a, F, L> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut out = if self.complete {
            String::from(
                r#"
\documentclass[margin=10pt]{standalone}
\usepackage{tikz,forest}
\usetikzlibrary{arrows.meta}"#,
            )
        } else {
            String::from("")
        };
        out.push_str(
            r#"
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
}"#,
        );

        if self.complete {
            out.push_str(r#"\begin{document}"#);
        }
        out.push_str(r#"\begin{forest}"#);

        out.push_str(&self.format_node(self.tree.root_node()));
        out.push_str(&self.legend());
        out.push_str("\n\t\\end{forest}\n");
        if self.complete {
            out.push_str("\\end{document}");
        }

        write!(f, "{}", out)
    }
}
