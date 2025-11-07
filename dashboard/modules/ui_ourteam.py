# ui_team.py
import streamlit as st

def render():
    st.markdown("""
        <style>
        .team-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px;margin-top:1rem}
        .card{background:var(--background-color,#ffffff);border:1px solid rgba(0,0,0,.08);border-radius:16px;padding:14px;box-shadow:0 2px 8px rgba(0,0,0,.04);text-align:center}
        .card img{width:120px;height:120px;object-fit:cover;border-radius:999px;display:block;margin:0 auto 10px;border:3px solid rgba(0,0,0,.06)}
        .role{font-size:.85rem;color:#6b7280;margin-top:2px}
        .quote{font-size:.95rem;color:#374151;line-height:1.35;margin-top:8px}
        .section{background:rgba(0,0,0,.02);border:1px solid rgba(0,0,0,.06);border-radius:16px;padding:16px}
        .kicker{letter-spacing:.08em;text-transform:uppercase;font-weight:600;font-size:.8rem;color:#6b7280;margin-bottom:6px}
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("Our team")
    st.write("Get to know the people behind our team, and discover our mission and the goals that guide everything we do")

    # Team data (cambia nombres/fotos si quieres). Las imágenes usan placeholders públicos.
    team = [
        {
            "name": "A. Analyst",
            "img": "https://placehold.co/400x400/png?text=A",
            "role": "Data Scientist",
            "quote": "Predictive models turn scattered incidents into early-warning signals."
        },
        {
            "name": "B. Builder",
            "img": "https://placehold.co/400x400/png?text=B",
            "role": "ML Engineer",
            "quote": "Forecasting crime risk helps allocate resources before spikes happen."
        },
        {
            "name": "C. Cartographer",
            "img": "https://placehold.co/400x400/png?text=C",
            "role": "Geo Analyst",
            "quote": "Space-time patterns reveal where prevention can be most effective."
        },
        {
            "name": "D. Detective",
            "img": "https://placehold.co/400x400/png?text=D",
            "role": "Research Lead",
            "quote": "From noise to narrative: modeling explains the ‘why’, not just the ‘what’."
        },
        {
            "name": "E. Ethicist",
            "img": "https://placehold.co/400x400/png?text=E",
            "role": "Policy & Ethics",
            "quote": "Responsible prediction means insight that informs action—never bias."
        },
    ]

    # Team grid
    st.markdown('<div class="kicker">Team</div>', unsafe_allow_html=True)
    st.markdown('<div class="team-grid">', unsafe_allow_html=True)
    for m in team:
        st.markdown(
            f"""
            <div class="card">
                <img src="{m['img']}" alt="{m['name']}">
                <div style="font-weight:700">{m['name']}</div>
                <div class="role">{m['role']}</div>
                <div class="quote">“{m['quote']}”</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")  # spacing

    # Mission
    st.markdown('<div class="kicker">Our mission</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section">
        <p><strong>Our mission is to transform raw data into knowledge that drives meaningful change.</strong>
        In a city where hundreds of crimes are reported every day, understanding the when, where, and why
        behind each incident is essential. We believe that data, when analyzed with purpose and precision,
        can illuminate the patterns that shape urban safety and help guide smarter decisions.</p>

        <p>Through the use of data analytics, visualization, and social insight, our goal is to uncover the stories
        hidden within the numbers — revealing how daily routines, social behavior, and city structure influence crime
        dynamics. By doing so, we aim to support evidence-based prevention strategies that make our communities safer
        and more resilient.</p>

        <p>We stand by the idea that information alone is not enough — it must lead to understanding, and understanding
        must lead to action. That is the path we follow: moving from incidents to insights.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")  # spacing

    # Goals
    st.markdown('<div class="kicker">Our goals</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section">
        <p><strong>Our goal is to turn complex crime data into clear, actionable insights that help understand and prevent urban insecurity.</strong>
        We aim to combine technology, analytical thinking, and social awareness to identify meaningful trends and support
        decision-makers, researchers, and communities in developing effective strategies for safety and prevention.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
