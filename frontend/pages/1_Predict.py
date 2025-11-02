import os
import sys
from pathlib import Path
from typing import List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is importable when Streamlit sets CWD to this folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.utils.model import get_paths, list_class_names, load_tf_model, prepare_image_batch
from frontend.utils.gradcam import compute_gradcam
from frontend.utils.guidance import GUIDANCE


def render_header() -> None:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2.5rem; padding: 1.5rem 0;">
        <h1 style="color: var(--text-primary); margin-bottom: 0.75rem; font-weight: 600; letter-spacing: -0.025em;">Plant Disease Prediction</h1>
        <p style="color: var(--text-secondary); font-size: 1.125rem; margin: 0; max-width: 700px; margin-left: auto; margin-right: auto;">Upload leaf images for instant AI-powered disease detection and expert guidance</p>
    </div>
    """, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Predict · Plant Pulse", page_icon="🧪", layout="wide")
    render_header()

    st.markdown("<div class='app-header'><span class='app-badge'>AI-Powered</span></div>", unsafe_allow_html=True)

    tab_upload, tab_history, tab_charts, tab_calculator = st.tabs(["Upload & Predict", "Analysis History", "Interactive Charts", "Cost Calculator"]) 

    # Lazily load the model only when needed so that upload/camera
    # widgets render immediately and the UI doesn't block on load.
    model = st.session_state.get("_cached_model")

    model_path, train_dir = get_paths()
    classes = list_class_names(train_dir)

    with tab_upload:
        left, right = st.columns((5, 7), gap="large")
        with left:
            st.markdown("### Upload Images")
            st.markdown("""
            <div class="upload-area">
                <p style="margin: 0; color: var(--text-secondary); font-size: 0.95rem;">Drag and drop your leaf images here</p>
                <p style="margin: 0.5rem 0 0; font-size: 0.875rem; color: var(--text-tertiary);">Supports JPG, PNG formats</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "Choose files", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="collapsed"
            )
            
            st.markdown("### Capture Live")
            # Camera can be blocked by browser permissions; render defensively
            camera = st.camera_input("Take a photo with your camera", label_visibility="collapsed")
            if camera is not None:
                uploaded_files = list(uploaded_files or []) + [camera]
            else:
                with st.expander("Camera not working?", expanded=False):
                    st.markdown(
                        "- Allow camera permissions in your browser settings.\n"
                        "- On Windows, ensure no other app is using the camera.\n"
                        "- If running remotely, use HTTPS or localhost to access camera."
                    )

            gallery_placeholder = st.empty()

            if uploaded_files:
                images = [Image.open(f) for f in uploaded_files]
                st.markdown("### Preview")
                gallery_placeholder.image(
                    images,
                    caption=[getattr(f, "name", "camera.jpg") for f in uploaded_files],
                    use_container_width=True,
                )
            
            st.markdown("### Analyze")
            if st.button("Run AI Analysis", type="primary", use_container_width=True) and uploaded_files:
                # Load model on first use
                if model is None:
                    try:
                        with st.spinner("Loading model..."):
                            model = load_tf_model()
                            st.session_state["_cached_model"] = model
                    except Exception as e:  # pragma: no cover
                        st.error(str(e))
                        st.stop()

                batch = prepare_image_batch(images)
                probs = model.predict(batch, verbose=0)

                records = []
                for i, p in enumerate(probs):
                    top3_idx = np.argsort(p)[-3:][::-1]
                    top_label = classes[int(top3_idx[0])] if top3_idx[0] < len(classes) else str(int(top3_idx[0]))
                    st.session_state["last_prediction_label"] = top_label
                    st.session_state["last_prediction_probs"] = p.tolist()
                    st.session_state["class_names"] = classes
                    # store history
                    hist = st.session_state.get("history", [])
                    hist.append({
                        "file": getattr(uploaded_files[i], "name", f"camera_{i}.jpg"),
                        "label": top_label,
                        "probs": p.tolist(),
                    })
                    st.session_state["history"] = hist

                    records.append(
                        {
                            "file": getattr(uploaded_files[i], "name", f"camera_{i}.jpg"),
                            "top1": top_label,
                            "confidence": float(p[int(top3_idx[0])]),
                        }
                    )

                df = pd.DataFrame(records)
                with right:
                    st.markdown("### Analysis Results")
                    st.markdown("""
                    <div class="results-card">
                        <h4 style="color: var(--primary-light); margin-bottom: 1rem; font-weight: 600;">Disease Detection Results</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(df, hide_index=True, use_container_width=True)

                    # Pick which image to inspect in detail
                    selected_idx = st.selectbox(
                        "Inspect details for image",
                        options=list(range(len(images))),
                        index=0,
                        format_func=lambda i: getattr(uploaded_files[i], "name", f"camera_{i}.jpg"),
                    )

                    # Human‑readable top prediction for the selected image
                    p_sel = probs[selected_idx]
                    top3_sel = np.argsort(p_sel)[-3:][::-1]
                    top_idx_sel = int(top3_sel[0])
                    top_label_sel = classes[top_idx_sel] if top_idx_sel < len(classes) else str(top_idx_sel)
                    confidence_sel = float(p_sel[top_idx_sel])
                    pretty_label = top_label_sel.replace("___", " → ").replace("_", " ")

                    st.success(
                        f"**Prediction:** {pretty_label}  •  **Confidence:** {confidence_sel:.1%}"
                    )

                    # Show Top‑3 classes for the selected image
                    st.markdown("### Top 3 Candidates")
                    for rank, idx in enumerate(top3_sel, start=1):
                        name = classes[int(idx)] if int(idx) < len(classes) else str(int(idx))
                        name = name.replace("___", " → ").replace("_", " ")
                        confidence_pct = float(p_sel[int(idx)])
                        st.markdown(f"**{rank}.** {name} — {confidence_pct:.1%}")

                    # Interactive Chart for selected image
                    p = p_sel
                    chart_df = pd.DataFrame({"class": classes, "prob": p})
                    # Clean up class names for display
                    chart_df["class_display"] = chart_df["class"].str.replace("___", " → ").str.replace("_", " ")
                    # Sort by probability
                    chart_df = chart_df.sort_values("prob", ascending=False).head(10)  # Top 10
                    
                    st.markdown("#### Confidence Distribution")
                    chart = (
                        alt.Chart(chart_df)
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#16a34a")
                        .encode(
                            x=alt.X("prob:Q", axis=alt.Axis(format=".0%", title="Confidence"), scale=alt.Scale(domain=[0, 1])), 
                            y=alt.Y("class_display:N", sort="-x", title="Disease Type"),
                            tooltip=[
                                alt.Tooltip("class_display:N", title="Disease"),
                                alt.Tooltip("prob:Q", title="Confidence", format=".1%")
                            ]
                        )
                        .properties(height=min(400, max(200, len(chart_df) * 30)))
                        .interactive()
                    )
                    st.altair_chart(chart, use_container_width=True)

                    # Grad-CAM visualizations for all images
                    st.markdown("#### AI Visual Insights")
                    st.markdown("**Grad-CAM Visualization:** Red areas indicate regions where the AI detected disease indicators")
                    
                    import matplotlib.cm as cm
                    
                    # Generate Grad-CAM for all images
                    with st.spinner("Generating visual insights for all images..."):
                        for img_idx in range(len(images)):
                            img = images[img_idx]
                            img_prob = probs[img_idx]
                            top_idx_img = int(np.argmax(img_prob))
                            top_label_img = classes[top_idx_img] if top_idx_img < len(classes) else str(top_idx_img)
                            pretty_label_img = top_label_img.replace("___", " → ").replace("_", " ")
                            confidence_img = float(img_prob[top_idx_img])
                            
                            # Get image name
                            img_name = getattr(uploaded_files[img_idx], "name", f"camera_{img_idx}.jpg")
                            
                            # Create Grad-CAM visualization
                            arr = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
                            heat = compute_gradcam(model, arr, top_idx_img)
                            colored = (cm.jet(heat)[..., :3] * 255).astype(np.uint8)
                            overlay = Image.fromarray(colored).resize(img.size)
                            base = img.convert("RGBA").copy()
                            overlay_rgba = overlay.convert("RGBA")
                            overlay_rgba.putalpha(120)
                            base.paste(overlay_rgba, (0, 0), overlay_rgba)
                            
                            # Display with prediction info
                            st.markdown(f"**{img_name}** — {pretty_label_img} ({confidence_img:.1%})")
                            st.image(
                                [img, base],
                                caption=["Original Image", "AI Heat Map"],
                                use_container_width=True,
                            )
                            
                            if img_idx < len(images) - 1:
                                st.divider()

                    # Guidance card
                    top_label = df.iloc[0]["top1"]
                    g = GUIDANCE.get(top_label)
                    if g:
                        st.markdown("#### Expert Guidance")
                        st.markdown("""
                        <div class="results-card">
                            <h4 style="color: var(--primary-light); margin-bottom: 1rem; font-weight: 600;">Treatment & Prevention</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        for section, bullets in g.items():
                            st.markdown(f"**{section}**")
                            for b in bullets:
                                st.markdown(f"• {b}")

                    st.download_button(
                        "Export Results (CSV)",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="plant_pulse_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    st.divider()
                    st.markdown("### Need More Help?")
                    st.page_link("pages/2_Chat.py", label="Ask follow-up questions in Chat", use_container_width=True)
        with right:
            if not uploaded_files:
                st.markdown("""
                <div style="text-align: center; padding: 3rem 2rem; background: var(--bg-primary); border-radius: var(--radius-xl); border: 2px dashed var(--border);">
                    <h3 style="color: var(--text-primary); margin-bottom: 1rem; font-weight: 600;">Ready to Analyze</h3>
                    <p style="color: var(--text-tertiary); margin: 0; line-height: 1.6;">Upload leaf images to get instant AI-powered disease detection, visual explanations, and expert guidance.</p>
                </div>
                """, unsafe_allow_html=True)

    with tab_history:
        st.markdown("### Analysis History")
        history = st.session_state.get("history", [])
        if not history:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 2rem; background: var(--bg-primary); border-radius: var(--radius-xl); border: 2px dashed var(--border);">
                <h3 style="color: var(--text-primary); margin-bottom: 1rem; font-weight: 600;">No Analysis Yet</h3>
                <p style="color: var(--text-tertiary); margin: 0;">Run your first prediction to see the history here.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("#### Previous Analysis Results")
            dfh = pd.DataFrame([
                {"file": h["file"], "label": h["label"], "top_prob": float(np.max(h["probs"]))}
                for h in history
            ])
            st.dataframe(dfh, hide_index=True, use_container_width=True)
            st.download_button(
                "Download History (CSV)",
                data=dfh.to_csv(index=False).encode("utf-8"),
                file_name="plant_pulse_history.csv",
                mime="text/csv",
                use_container_width=True
            )

    with tab_charts:
        st.markdown("### Interactive Analytics Dashboard")
        st.markdown("Visualize disease patterns and trends from your analysis history")
        
        history = st.session_state.get("history", [])
        
        if not history:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 2rem; background: var(--bg-primary); border-radius: var(--radius-xl); border: 2px dashed var(--border);">
                <h3 style="color: var(--text-primary); margin-bottom: 1rem; font-weight: 600;">No Data Available</h3>
                <p style="color: var(--text-tertiary); margin: 0;">Run some predictions to see interactive charts and analytics here.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Prepare data
            chart_data = []
            for i, h in enumerate(history):
                label = h["label"].replace("___", " → ").replace("_", " ")
                chart_data.append({
                    "index": i,
                    "disease": label,
                    "confidence": float(np.max(h["probs"])),
                    "file": h["file"]
                })
            
            df_charts = pd.DataFrame(chart_data)
            
            if len(df_charts) == 0:
                st.warning("No data available for charts.")
            else:
                # Debug: Show data preview (can be removed later)
                with st.expander("📊 Data Preview (Debug)", expanded=False):
                    st.dataframe(df_charts.head(10))
                    st.write(f"Total records: {len(df_charts)}")
                    st.write(f"Unique diseases: {df_charts['disease'].nunique()}")
                # Chart 1: Disease Distribution
                col1, col2 = st.columns(2, gap="large")
                
                with col1:
                    st.markdown("#### Disease Frequency")
                    # Get value counts and ensure proper column names
                    vc = df_charts["disease"].value_counts()
                    disease_counts = pd.DataFrame({
                        "disease": vc.index.tolist(),
                        "count": vc.values.tolist()
                    })
                    
                    # Ensure we have data
                    if len(disease_counts) > 0 and disease_counts["count"].sum() > 0:
                        max_count = disease_counts["count"].max()
                        chart_freq = (
                            alt.Chart(disease_counts)
                            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#3b82f6")
                            .encode(
                                x=alt.X("count:Q", title="Frequency", scale=alt.Scale(domain=[0, max(1, max_count * 1.2)], nice=True)),
                                y=alt.Y("disease:N", sort="-x", title="Disease Type"),
                                tooltip=[
                                    alt.Tooltip("disease:N", title="Disease"),
                                    alt.Tooltip("count:Q", title="Count", format=".0f")
                                ],
                                color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None)
                            )
                            .properties(height=max(300, min(600, len(disease_counts) * 50)))
                            .configure_axis(grid=True)
                            .configure_view(strokeWidth=0)
                        )
                        st.altair_chart(chart_freq, use_container_width=True)
                    else:
                        st.info("No disease data to display.")
                
                with col2:
                    st.markdown("#### Average Confidence by Disease")
                    avg_conf = df_charts.groupby("disease")["confidence"].mean().reset_index()
                    avg_conf.columns = ["disease", "avg_confidence"]
                    avg_conf = avg_conf.sort_values("avg_confidence", ascending=False)
                    
                    # Ensure we have data
                    if len(avg_conf) > 0 and avg_conf["avg_confidence"].notna().any():
                        chart_avg = (
                            alt.Chart(avg_conf)
                            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#22c55e")
                            .encode(
                                x=alt.X("avg_confidence:Q", axis=alt.Axis(format=".0%", title="Avg Confidence"), scale=alt.Scale(domain=[0, 1], nice=True)),
                                y=alt.Y("disease:N", sort="-x", title="Disease Type"),
                                tooltip=[
                                    alt.Tooltip("disease:N", title="Disease"),
                                    alt.Tooltip("avg_confidence:Q", title="Avg Confidence", format=".1%")
                                ],
                                color=alt.Color("avg_confidence:Q", scale=alt.Scale(scheme="greens"), legend=None)
                            )
                            .properties(height=max(300, min(600, len(avg_conf) * 50)))
                            .configure_axis(grid=True)
                            .configure_view(strokeWidth=0)
                        )
                        st.altair_chart(chart_avg, use_container_width=True)
                    else:
                        st.info("No confidence data to display.")
            
                # Chart 2: Confidence Trends Over Time
                st.markdown("#### Confidence Trends")
                df_trends = df_charts.copy()
                df_trends["analysis_num"] = range(len(df_trends))
                
                if len(df_trends) > 0:
                    chart_trend = (
                        alt.Chart(df_trends)
                        .mark_line(point=True, strokeWidth=3)
                        .encode(
                            x=alt.X("analysis_num:Q", title="Analysis Number", axis=alt.Axis(tickCount=min(10, len(df_trends)))),
                            y=alt.Y("confidence:Q", axis=alt.Axis(format=".0%", title="Confidence"), scale=alt.Scale(domain=[0, 1])),
                            tooltip=[
                                alt.Tooltip("analysis_num:Q", title="Analysis #", format=".0f"),
                                alt.Tooltip("disease:N", title="Disease"),
                                alt.Tooltip("confidence:Q", title="Confidence", format=".1%"),
                                alt.Tooltip("file:N", title="File")
                            ],
                            color=alt.Color("disease:N", scale=alt.Scale(scheme="category20"), legend=alt.Legend(title="Disease", orient="bottom"))
                        )
                        .properties(height=350)
                        .configure_axis(grid=True)
                        .interactive()
                    )
                    st.altair_chart(chart_trend, use_container_width=True)
                else:
                    st.info("No trend data available.")
                
                # Chart 3: Scatter Plot - Disease vs Confidence
                st.markdown("#### Disease Distribution Analysis")
                if len(df_charts) > 0:
                    chart_scatter = (
                        alt.Chart(df_charts)
                        .mark_circle(size=100, opacity=0.6)
                        .encode(
                            x=alt.X("disease:N", title="Disease Type", axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y("confidence:Q", axis=alt.Axis(format=".0%", title="Confidence"), scale=alt.Scale(domain=[0, 1])),
                            size=alt.Size("confidence:Q", scale=alt.Scale(range=[50, 300]), legend=alt.Legend(title="Confidence")),
                            color=alt.Color("disease:N", scale=alt.Scale(scheme="category20"), legend=alt.Legend(title="Disease")),
                            tooltip=[
                                alt.Tooltip("disease:N", title="Disease"),
                                alt.Tooltip("confidence:Q", title="Confidence", format=".1%"),
                                alt.Tooltip("file:N", title="File")
                            ]
                        )
                        .properties(height=400)
                        .configure_axis(grid=True)
                        .interactive()
                    )
                    st.altair_chart(chart_scatter, use_container_width=True)
                else:
                    st.info("No distribution data available.")
                
                # Summary Stats
                st.markdown("#### Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Analyses", len(df_charts))
                with col2:
                    st.metric("Unique Diseases", df_charts["disease"].nunique())
                with col3:
                    st.metric("Avg Confidence", f"{df_charts['confidence'].mean():.1%}")
                with col4:
                    st.metric("Highest Confidence", f"{df_charts['confidence'].max():.1%}")

    with tab_calculator:
        st.markdown("### Treatment Cost Calculator")
        st.markdown("Estimate treatment costs based on detected diseases")
        
        history = st.session_state.get("history", [])
        
        if not history:
            st.info("Run some predictions first to use the cost calculator. The calculator will use your most recent detection.")
            use_manual = True
            selected_disease = None
        else:
            use_manual = st.checkbox("Manually select disease", value=False)
            if use_manual:
                selected_disease = None
            else:
                # Use most recent detection
                most_recent = history[-1]
                selected_disease = most_recent["label"]
                disease_display = selected_disease.replace("___", " → ").replace("_", " ")
                st.markdown(f"""
                <div style="background: var(--bg-primary); border: 1px solid var(--border); 
                            border-left: 4px solid var(--primary); border-radius: var(--radius-lg); 
                            padding: 1rem; margin-bottom: 1.5rem; box-shadow: var(--shadow-sm);">
                    <p style="margin: 0; color: var(--text-secondary); font-size: 0.9375rem;">
                        <strong style="color: var(--primary-light); font-weight: 600;">Detected Disease:</strong> 
                        <span>{disease_display}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        if use_manual or not history:
            # Disease selector
            if not history:
                disease_options = [d.replace("___", " → ").replace("_", " ") for d in classes]
            else:
                unique_diseases = list(set([h["label"] for h in history]))
                disease_options = [d.replace("___", " → ").replace("_", " ") for d in unique_diseases] + \
                                 [d.replace("___", " → ").replace("_", " ") for d in classes if d not in unique_diseases]
            
            selected_disease_display = st.selectbox(
                "Select Disease",
                options=disease_options,
                index=0 if not disease_options else None
            )
            
            # Convert back to internal format
            if selected_disease_display:
                selected_disease = selected_disease_display.replace(" → ", "___").replace(" ", "_")
                # Find matching class
                for cls in classes:
                    if cls.replace("___", " → ").replace("_", " ") == selected_disease_display:
                        selected_disease = cls
                        break
        
        if selected_disease:
            st.divider()
            
            # Input parameters
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown("#### Treatment Parameters")
                treatment_type = st.radio(
                    "Treatment Type",
                    options=["Organic", "Chemical", "Integrated"],
                    horizontal=True,
                    help="Organic: natural treatments, Chemical: synthetic pesticides, Integrated: combination approach"
                )
                
                scale_type = st.selectbox(
                    "Scale Type",
                    options=["Acres", "Plants", "Square Meters", "Hectares"]
                )
                
                if scale_type == "Acres":
                    scale = st.number_input("Area (Acres)", min_value=0.01, max_value=1000.0, value=1.0, step=0.1)
                elif scale_type == "Plants":
                    scale = st.number_input("Number of Plants", min_value=1, max_value=10000, value=10, step=1)
                elif scale_type == "Square Meters":
                    scale = st.number_input("Area (Square Meters)", min_value=1, max_value=100000, value=100, step=1)
                else:  # Hectares
                    scale = st.number_input("Area (Hectares)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
            
            with col2:
                st.markdown("#### Cost Parameters")
                labor_cost_per_hour = st.number_input(
                    "Labor Cost per Hour ($)", 
                    min_value=0.0, 
                    max_value=200.0, 
                    value=25.0, 
                    step=1.0,
                    help="Hourly rate for treatment application"
                )
                
                material_cost_multiplier = st.slider(
                    "Material Cost Factor", 
                    min_value=0.5, 
                    max_value=3.0, 
                    value=1.0, 
                    step=0.1,
                    help="Adjustment factor for material costs (1.0 = standard pricing)"
                )
            
            # Calculate costs
            # Base cost estimates (per unit area/plant)
            disease_cost_base = {
                "Apple___Apple_scab": {"organic": 50, "chemical": 30, "integrated": 40},
                "Apple___Black_rot": {"organic": 60, "chemical": 35, "integrated": 45},
                "Apple___Cedar_apple_rust": {"organic": 55, "chemical": 32, "integrated": 42},
            }
            
            # Default cost if disease not in dictionary
            default_cost = {"organic": 55, "chemical": 32, "integrated": 42}
            
            treatment_key = treatment_type.lower()
            base_cost_per_unit = disease_cost_base.get(selected_disease, default_cost).get(treatment_key, 50)
            
            # Calculate material costs
            if scale_type == "Acres":
                material_cost = base_cost_per_unit * scale * material_cost_multiplier
                labor_hours = scale * 2  # ~2 hours per acre
            elif scale_type == "Plants":
                cost_per_plant = base_cost_per_unit / 10  # Assuming ~10 plants per unit area
                material_cost = cost_per_plant * scale * material_cost_multiplier
                labor_hours = (scale / 10) * 2  # ~2 hours per 10 plants
            elif scale_type == "Square Meters":
                # Convert to acres equivalent (1 acre ≈ 4047 m²)
                acres_equivalent = scale / 4047
                material_cost = base_cost_per_unit * acres_equivalent * material_cost_multiplier
                labor_hours = acres_equivalent * 2
            else:  # Hectares
                # 1 hectare ≈ 2.47 acres
                acres_equivalent = scale * 2.47
                material_cost = base_cost_per_unit * acres_equivalent * material_cost_multiplier
                labor_hours = acres_equivalent * 2
            
            labor_cost = labor_hours * labor_cost_per_hour
            total_cost = material_cost + labor_cost
            
            # Display results
            st.divider()
            st.markdown("#### Cost Breakdown")
            
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            
            with result_col1:
                st.metric("Material Cost", f"${material_cost:,.2f}")
            with result_col2:
                st.metric("Labor Cost", f"${labor_cost:,.2f}")
            with result_col3:
                st.metric("Total Cost", f"${total_cost:,.2f}")
            with result_col4:
                st.metric("Est. Hours", f"{labor_hours:.1f}")
            
            # Cost visualization
            cost_data = pd.DataFrame({
                "Category": ["Material", "Labor"],
                "Cost": [material_cost, labor_cost]
            })
            
            chart_cost = (
                alt.Chart(cost_data)
                .mark_arc(innerRadius=50, stroke="#fff")
                .encode(
                    theta=alt.Theta("Cost:Q", stack=True),
                    color=alt.Color("Category:N", scale=alt.Scale(domain=["Material", "Labor"], range=["#16a34a", "#3b82f6"]), legend=alt.Legend(title="Cost Type")),
                    tooltip=[
                        alt.Tooltip("Category:N", title="Category"),
                        alt.Tooltip("Cost:Q", title="Cost", format="$,.2f")
                    ]
                )
                .properties(height=300, title="Cost Distribution")
            )
            
            st.altair_chart(chart_cost, use_container_width=True)
            
            # Treatment recommendations
            g = GUIDANCE.get(selected_disease)
            if g:
                st.markdown("#### Treatment Recommendations")
                st.markdown("""
                <div class="results-card">
                    <h4 style="color: var(--primary-light); margin-bottom: 1rem; font-weight: 600;">Management Strategy</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if "Management" in g:
                    st.markdown("**Management Actions:**")
                    for item in g["Management"]:
                        st.markdown(f"• {item}")
                
                if "Prevention" in g:
                    st.markdown("**Prevention Tips:**")
                    for item in g["Prevention"]:
                        st.markdown(f"• {item}")
            
            # Notes
            st.info("""
            **Note:** These are estimated costs based on typical treatment scenarios. Actual costs may vary based on:
            - Regional pricing differences
            - Specific product selection
            - Severity of disease
            - Additional equipment needs
            - Multiple treatment cycles required
            """)


if __name__ == "__main__":
    main()


