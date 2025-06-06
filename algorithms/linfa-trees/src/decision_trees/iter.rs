use std::collections::VecDeque;
use std::fmt::Debug;
use std::iter::Iterator;

use super::TreeNode;
use linfa::{Float, Label};

/// Level-order (BFT) iterator of nodes in a decision tree
#[derive(Debug, Clone, PartialEq)]
pub struct NodeIter<'a, F, L> {
    queue: VecDeque<&'a TreeNode<F, L>>,
}

impl<'a, F, L> NodeIter<'a, F, L> {
    pub fn new(queue: VecDeque<&'a TreeNode<F, L>>) -> Self {
        NodeIter { queue }
    }
}

impl<'a, F: Float, L: Debug + Label> Iterator for NodeIter<'a, F, L> {
    type Item = &'a TreeNode<F, L>;

    fn next(&mut self) -> Option<Self::Item> {
        #[allow(clippy::manual_inspect)]
        self.queue.pop_front().map(|node| {
            node.children()
                .into_iter()
                .filter_map(|x| x.as_ref())
                .for_each(|child| self.queue.push_back(child));
            node
        })
    }
}
