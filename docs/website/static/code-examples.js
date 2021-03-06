function toggle_selected(t) {
    let e = t.value;
    document.querySelector(".example.visible").classList.remove("visible")
    document.querySelector(`.example[data-example="${e}"]`).classList.add("visible")
}

document.addEventListener("DOMContentLoaded", function() {
    let a = document.querySelector(".logo span")
    a.addEventListener("click", function(t) {
        let obj = document.querySelector(".header-main ul");

        if(obj.style.display === "" || obj.style.display === "none"){
            obj.style.display = "block";
        } else {
            obj.style.display = "none";
        }
    });

    let n = document.querySelector(".code-examples select")
    n.addEventListener("change", function(t) {
        toggle_selected(t.target)
    })

    let ops  = n.querySelectorAll("option")
    ops[Math.floor(Math.random() * ops.length)].selected = !0

    toggle_selected(n)
});
