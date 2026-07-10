// main.js - Nasdaq 100 Direction Predictor Interface Enhancements

document.addEventListener("DOMContentLoaded", () => {
    console.log("NQ.XGBoost UI Initialized.");

    // Cascading slide-up animations for the stock input cards
    const inputCards = document.querySelectorAll(".stock-input-card");
    inputCards.forEach((card, index) => {
        card.style.opacity = "0";
        card.style.transform = "translateY(15px)";
        
        setTimeout(() => {
            card.style.transition = "opacity 0.5s cubic-bezier(0.1, 0.8, 0.2, 1), transform 0.5s cubic-bezier(0.1, 0.8, 0.2, 1), border-color 0.2s ease, box-shadow 0.2s ease";
            card.style.opacity = "1";
            card.style.transform = "translateY(0)";
        }, index * 40); // 40ms stagger between cards
    });

    // Animate probability bars after load
    const bars = document.querySelectorAll(".prob-bar-fill");
    bars.forEach((bar) => {
        const targetWidth = bar.style.width;
        bar.style.width = "0%";
        setTimeout(() => {
            bar.style.width = targetWidth;
        }, 150);
    });
});
