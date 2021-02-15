function toggle_selected(t) {
    let e = t.value;
    console.log(e)
    console.log(document.querySelector(".example.visible"))
    document.querySelector(".example.visible").classList.remove("visible")
    document.querySelector(`.example[data-example="${e}"]`).classList.add("visible")
}

document.addEventListener("DOMContentLoaded", function() {
    let n = document.querySelector(".code-examples select")
    n.addEventListener("change", function(t) {
        toggle_selected(t.target)
    })

    let ops  = n.querySelectorAll("option")
    ops[Math.floor(Math.random() * ops.length)].selected = !0

    toggle_selected(n)
});
