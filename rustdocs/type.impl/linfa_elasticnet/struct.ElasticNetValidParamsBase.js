(function() {
    var type_impls = Object.fromEntries([["linfa_elasticnet",[["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Clone-for-ElasticNetValidParamsBase%3CF,+MULTI_TASK%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/linfa_elasticnet/hyperparams.rs.html#15\">Source</a><a href=\"#impl-Clone-for-ElasticNetValidParamsBase%3CF,+MULTI_TASK%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.85.1/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>, const MULTI_TASK: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.85.1/std/primitive.bool.html\">bool</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.85.1/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a> for <a class=\"struct\" href=\"linfa_elasticnet/struct.ElasticNetValidParamsBase.html\" title=\"struct linfa_elasticnet::ElasticNetValidParamsBase\">ElasticNetValidParamsBase</a>&lt;F, MULTI_TASK&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.clone\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/linfa_elasticnet/hyperparams.rs.html#15\">Source</a><a href=\"#method.clone\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.85.1/core/clone/trait.Clone.html#tymethod.clone\" class=\"fn\">clone</a>(&amp;self) -&gt; <a class=\"struct\" href=\"linfa_elasticnet/struct.ElasticNetValidParamsBase.html\" title=\"struct linfa_elasticnet::ElasticNetValidParamsBase\">ElasticNetValidParamsBase</a>&lt;F, MULTI_TASK&gt;</h4></section></summary><div class='docblock'>Returns a copy of the value. <a href=\"https://doc.rust-lang.org/1.85.1/core/clone/trait.Clone.html#tymethod.clone\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.clone_from\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/1.85.1/src/core/clone.rs.html#174\">Source</a></span><a href=\"#method.clone_from\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.85.1/core/clone/trait.Clone.html#method.clone_from\" class=\"fn\">clone_from</a>(&amp;mut self, source: &amp;Self)</h4></section></summary><div class='docblock'>Performs copy-assignment from <code>source</code>. <a href=\"https://doc.rust-lang.org/1.85.1/core/clone/trait.Clone.html#method.clone_from\">Read more</a></div></details></div></details>","Clone","linfa_elasticnet::hyperparams::ElasticNetValidParams","linfa_elasticnet::hyperparams::MultiTaskElasticNetValidParams"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Debug-for-ElasticNetValidParamsBase%3CF,+MULTI_TASK%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/linfa_elasticnet/hyperparams.rs.html#15\">Source</a><a href=\"#impl-Debug-for-ElasticNetValidParamsBase%3CF,+MULTI_TASK%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.85.1/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a>, const MULTI_TASK: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.85.1/std/primitive.bool.html\">bool</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.85.1/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a> for <a class=\"struct\" href=\"linfa_elasticnet/struct.ElasticNetValidParamsBase.html\" title=\"struct linfa_elasticnet::ElasticNetValidParamsBase\">ElasticNetValidParamsBase</a>&lt;F, MULTI_TASK&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.fmt\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/linfa_elasticnet/hyperparams.rs.html#15\">Source</a><a href=\"#method.fmt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.85.1/core/fmt/trait.Debug.html#tymethod.fmt\" class=\"fn\">fmt</a>(&amp;self, f: &amp;mut <a class=\"struct\" href=\"https://doc.rust-lang.org/1.85.1/core/fmt/struct.Formatter.html\" title=\"struct core::fmt::Formatter\">Formatter</a>&lt;'_&gt;) -&gt; <a class=\"type\" href=\"https://doc.rust-lang.org/1.85.1/core/fmt/type.Result.html\" title=\"type core::fmt::Result\">Result</a></h4></section></summary><div class='docblock'>Formats the value using the given formatter. <a href=\"https://doc.rust-lang.org/1.85.1/core/fmt/trait.Debug.html#tymethod.fmt\">Read more</a></div></details></div></details>","Debug","linfa_elasticnet::hyperparams::ElasticNetValidParams","linfa_elasticnet::hyperparams::MultiTaskElasticNetValidParams"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-ElasticNetValidParamsBase%3CF,+MULTI_TASK%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/linfa_elasticnet/hyperparams.rs.html#34-54\">Source</a><a href=\"#impl-ElasticNetValidParamsBase%3CF,+MULTI_TASK%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;F: Float, const MULTI_TASK: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.85.1/std/primitive.bool.html\">bool</a>&gt; <a class=\"struct\" href=\"linfa_elasticnet/struct.ElasticNetValidParamsBase.html\" title=\"struct linfa_elasticnet::ElasticNetValidParamsBase\">ElasticNetValidParamsBase</a>&lt;F, MULTI_TASK&gt;</h3></section></summary><div class=\"impl-items\"><section id=\"method.penalty\" class=\"method\"><a class=\"src rightside\" href=\"src/linfa_elasticnet/hyperparams.rs.html#35-37\">Source</a><h4 class=\"code-header\">pub fn <a href=\"linfa_elasticnet/struct.ElasticNetValidParamsBase.html#tymethod.penalty\" class=\"fn\">penalty</a>(&amp;self) -&gt; F</h4></section><section id=\"method.l1_ratio\" class=\"method\"><a class=\"src rightside\" href=\"src/linfa_elasticnet/hyperparams.rs.html#39-41\">Source</a><h4 class=\"code-header\">pub fn <a href=\"linfa_elasticnet/struct.ElasticNetValidParamsBase.html#tymethod.l1_ratio\" class=\"fn\">l1_ratio</a>(&amp;self) -&gt; F</h4></section><section id=\"method.with_intercept\" class=\"method\"><a class=\"src rightside\" href=\"src/linfa_elasticnet/hyperparams.rs.html#43-45\">Source</a><h4 class=\"code-header\">pub fn <a href=\"linfa_elasticnet/struct.ElasticNetValidParamsBase.html#tymethod.with_intercept\" class=\"fn\">with_intercept</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.85.1/std/primitive.bool.html\">bool</a></h4></section><section id=\"method.max_iterations\" class=\"method\"><a class=\"src rightside\" href=\"src/linfa_elasticnet/hyperparams.rs.html#47-49\">Source</a><h4 class=\"code-header\">pub fn <a href=\"linfa_elasticnet/struct.ElasticNetValidParamsBase.html#tymethod.max_iterations\" class=\"fn\">max_iterations</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.85.1/std/primitive.u32.html\">u32</a></h4></section><section id=\"method.tolerance\" class=\"method\"><a class=\"src rightside\" href=\"src/linfa_elasticnet/hyperparams.rs.html#51-53\">Source</a><h4 class=\"code-header\">pub fn <a href=\"linfa_elasticnet/struct.ElasticNetValidParamsBase.html#tymethod.tolerance\" class=\"fn\">tolerance</a>(&amp;self) -&gt; F</h4></section></div></details>",0,"linfa_elasticnet::hyperparams::ElasticNetValidParams","linfa_elasticnet::hyperparams::MultiTaskElasticNetValidParams"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-PartialEq-for-ElasticNetValidParamsBase%3CF,+MULTI_TASK%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/linfa_elasticnet/hyperparams.rs.html#15\">Source</a><a href=\"#impl-PartialEq-for-ElasticNetValidParamsBase%3CF,+MULTI_TASK%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.85.1/core/cmp/trait.PartialEq.html\" title=\"trait core::cmp::PartialEq\">PartialEq</a>, const MULTI_TASK: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.85.1/std/primitive.bool.html\">bool</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.85.1/core/cmp/trait.PartialEq.html\" title=\"trait core::cmp::PartialEq\">PartialEq</a> for <a class=\"struct\" href=\"linfa_elasticnet/struct.ElasticNetValidParamsBase.html\" title=\"struct linfa_elasticnet::ElasticNetValidParamsBase\">ElasticNetValidParamsBase</a>&lt;F, MULTI_TASK&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.eq\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/linfa_elasticnet/hyperparams.rs.html#15\">Source</a><a href=\"#method.eq\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.85.1/core/cmp/trait.PartialEq.html#tymethod.eq\" class=\"fn\">eq</a>(&amp;self, other: &amp;<a class=\"struct\" href=\"linfa_elasticnet/struct.ElasticNetValidParamsBase.html\" title=\"struct linfa_elasticnet::ElasticNetValidParamsBase\">ElasticNetValidParamsBase</a>&lt;F, MULTI_TASK&gt;) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.85.1/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>Tests for <code>self</code> and <code>other</code> values to be equal, and is used by <code>==</code>.</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.ne\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/1.85.1/src/core/cmp.rs.html#261\">Source</a></span><a href=\"#method.ne\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/1.85.1/core/cmp/trait.PartialEq.html#method.ne\" class=\"fn\">ne</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.85.1/std/primitive.reference.html\">&amp;Rhs</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.85.1/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>Tests for <code>!=</code>. The default implementation is almost always sufficient,\nand should not be overridden without very good reason.</div></details></div></details>","PartialEq","linfa_elasticnet::hyperparams::ElasticNetValidParams","linfa_elasticnet::hyperparams::MultiTaskElasticNetValidParams"],["<section id=\"impl-Eq-for-ElasticNetValidParamsBase%3CF,+MULTI_TASK%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/linfa_elasticnet/hyperparams.rs.html#15\">Source</a><a href=\"#impl-Eq-for-ElasticNetValidParamsBase%3CF,+MULTI_TASK%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;F: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.85.1/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a>, const MULTI_TASK: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.85.1/std/primitive.bool.html\">bool</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.85.1/core/cmp/trait.Eq.html\" title=\"trait core::cmp::Eq\">Eq</a> for <a class=\"struct\" href=\"linfa_elasticnet/struct.ElasticNetValidParamsBase.html\" title=\"struct linfa_elasticnet::ElasticNetValidParamsBase\">ElasticNetValidParamsBase</a>&lt;F, MULTI_TASK&gt;</h3></section>","Eq","linfa_elasticnet::hyperparams::ElasticNetValidParams","linfa_elasticnet::hyperparams::MultiTaskElasticNetValidParams"],["<section id=\"impl-StructuralPartialEq-for-ElasticNetValidParamsBase%3CF,+MULTI_TASK%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/linfa_elasticnet/hyperparams.rs.html#15\">Source</a><a href=\"#impl-StructuralPartialEq-for-ElasticNetValidParamsBase%3CF,+MULTI_TASK%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;F, const MULTI_TASK: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.85.1/std/primitive.bool.html\">bool</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.85.1/core/marker/trait.StructuralPartialEq.html\" title=\"trait core::marker::StructuralPartialEq\">StructuralPartialEq</a> for <a class=\"struct\" href=\"linfa_elasticnet/struct.ElasticNetValidParamsBase.html\" title=\"struct linfa_elasticnet::ElasticNetValidParamsBase\">ElasticNetValidParamsBase</a>&lt;F, MULTI_TASK&gt;</h3></section>","StructuralPartialEq","linfa_elasticnet::hyperparams::ElasticNetValidParams","linfa_elasticnet::hyperparams::MultiTaskElasticNetValidParams"]]]]);
    if (window.register_type_impls) {
        window.register_type_impls(type_impls);
    } else {
        window.pending_type_impls = type_impls;
    }
})()
//{"start":55,"fragment_lengths":[12497]}