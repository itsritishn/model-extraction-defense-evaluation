#!/usr/bin/env python3
"""
Interactive Web Dashboard for Model Extraction Defense Evaluation.

MSc Advanced Computer Science (Data Analytics) Dissertation
University of Leeds

Run with:
    streamlit run dashboard.py

Author: MSc Advanced Computer Science (Data Analytics) Dissertation
"""

import os
import time
import torch
import pandas as pd
import numpy as np
import streamlit as st
import torchvision.transforms as transforms
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# ══════════════════════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Model Extraction Defense Evaluation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# Import project modules
# ══════════════════════════════════════════════════════════════════════════════
try:
    from victim import VictimModel, build_victim_model, DEVICE
    from data_loader import (
        get_data_loaders,
        get_subset_loader,
        CIFAR10_CLASSES,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )
    from attack import run_attack, ATTACK_REGISTRY, build_substitute_model, train_substitute
    from defenses import get_all_defenses, get_defense_by_name
    from threat_model import ThreatConfig, QueryExecutor
    from evaluate import compute_all_metrics
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


# ══════════════════════════════════════════════════════════════════════════════
# Cached loaders
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_victim_model(arch: str = "resnet50"):
    """Load and cache the victim model."""
    # Map architecture names to checkpoint filenames
    arch_to_filename = {
        "resnet50": "victim_resnet50.pth",
        "efficientnet_b0": "victim_efficientnet.pth",
    }
    filename = arch_to_filename.get(arch, f"victim_{arch}.pth")
    checkpoint_path = f"models/{filename}"
    
    if not os.path.exists(checkpoint_path):
        return None, f"Checkpoint not found: {checkpoint_path}"
    
    model = build_victim_model(arch, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.eval()
    victim = VictimModel(model)
    return victim, None


@st.cache_resource
def load_data():
    """Load and cache CIFAR-10 data."""
    loaders = get_data_loaders(batch_size=64)
    return loaders["train"], loaders["test"]


@st.cache_data
def get_victim_accuracy(_victim, _test_loader, arch_name: str):
    """Compute victim accuracy (cached by arch_name)."""
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in _test_loader:
            images = images.to(DEVICE)
            probs = _victim.query(images)
            preds = probs.argmax(dim=1).cpu()  # Move to CPU for comparison
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def ensure_owner_defense_state() -> None:
    """Ensure server-side owner defense config exists in session state."""
    if "owner_defense_name" not in st.session_state:
        st.session_state.owner_defense_name = "none"
    if "owner_defense_fn" not in st.session_state:
        st.session_state.owner_defense_fn = get_defense_by_name(
            st.session_state.owner_defense_name
        )
    if "owner_victim_arch" not in st.session_state:
        st.session_state.owner_victim_arch = "resnet50"


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title("🛡️ Model Extraction")
    st.sidebar.markdown("**Defense Evaluation Dashboard**")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate to:",
        [
            "🏠 Overview",
            "🎯 Victim Model",
            "⚔️ Attack Demo",
            "🔒 Owner Settings",
            "🛡️ Defense Demo",
            "📊 Results Analysis",
            "📈 Visualizations",
        ],
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Info")
    st.sidebar.markdown(f"**Device:** `{DEVICE}`")
    st.sidebar.markdown(f"**PyTorch:** `{torch.__version__}`")
    if torch.cuda.is_available():
        st.sidebar.markdown(f"**GPU:** `{torch.cuda.get_device_name(0)}`")
    
    return page


# ══════════════════════════════════════════════════════════════════════════════
# Page: Overview
# ══════════════════════════════════════════════════════════════════════════════
def page_overview():
    """Render the overview page."""
    st.title("🛡️ Model Extraction Defense Evaluation")
    st.markdown("### MSc Advanced Computer Science (Data Analytics) Dissertation")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 The Problem")
        st.markdown("""
        Machine learning models deployed as APIs are vulnerable to 
        **model extraction attacks** where adversaries:
        
        1. Query the model with inputs
        2. Collect the output predictions
        3. Train a "substitute" model that mimics the original
        
        This threatens **intellectual property** and enables 
        **adversarial attacks**.
        """)
    
    with col2:
        st.markdown("### ⚔️ Attack Strategies")
        st.markdown("""
        We evaluate **3 attack strategies**:
        
        - **Random Query**: Query with random noise images
        - **Knockoff Nets**: Query with domain-relevant images
        - **Active Learning**: Iteratively select informative queries
        
        Using **2 substitute architectures**:
        - SmallCNN (~95K params)
        - MobileNetV3 (~1.5M params)
        """)
    
    with col3:
        st.markdown("### 🛡️ Defense Mechanisms")
        st.markdown("""
        We evaluate **3 proven-effective defenses** (weak defenses removed after empirical testing):
        
        1. 🟢 **Throttling** — Hard query limit; attacker gets uniform noise after budget exceeded
        2. 🟡 **Prediction Poisoning** — Randomly corrupt a fraction of returned probabilities
        3. 🟡 **Adaptive Noise** — Escalates noise level when suspicious query patterns detected
        """)
    
    st.markdown("---")
    
    # Pipeline diagram
    st.markdown("### 📐 Evaluation Pipeline")
    
    pipeline_fig = go.Figure()
    
    # Nodes
    nodes = ["Data\n(CIFAR-10)", "Victim\n(ResNet-50)", "Attacker\nQueries", 
             "Defense\nMechanism", "Substitute\nModel", "Metrics\nEvaluation"]
    x_pos = [0, 1, 2, 3, 4, 5]
    y_pos = [0, 0, 0, 0, 0, 0]
    
    pipeline_fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode='markers+text',
        marker=dict(size=60, color=['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#1abc9c']),
        text=nodes,
        textposition='bottom center',
        textfont=dict(size=12),
    ))
    
    # Arrows
    for i in range(len(x_pos) - 1):
        pipeline_fig.add_annotation(
            x=x_pos[i+1] - 0.15, y=0,
            ax=x_pos[i] + 0.15, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="#666",
        )
    
    pipeline_fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=200,
        margin=dict(l=20, r=20, t=20, b=60),
    )
    
    st.plotly_chart(pipeline_fig, use_container_width=True)
    
    # Key metrics
    st.markdown("### 📏 Evaluation Metrics")
    
    metrics_df = pd.DataFrame({
        "Metric": ["Fidelity", "Substitute Accuracy", "Protection Score", "Utility Cost"],
        "Description": [
            "% agreement between victim and substitute predictions",
            "Substitute model's accuracy on test set",
            "Reduction in extraction success due to defense",
            "Accuracy loss for legitimate users due to defense",
        ],
        "Goal": ["Lower = Better Defense", "Lower = Better Defense", 
                 "Higher = Better Defense", "Lower = Better Defense"],
    })
    
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page: Victim Model
# ══════════════════════════════════════════════════════════════════════════════
def page_victim_model():
    """Render the victim model page."""
    st.title("🎯 Victim Model")
    st.markdown("The target model that attackers want to extract.")
    
    st.markdown("---")
    
    # Model selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        arch = st.selectbox(
            "Select Victim Architecture:",
            ["resnet50"],
            index=0,
        )
        
        st.markdown("### Model Details")
        if arch == "resnet50":
            st.markdown("""
            - **Architecture:** ResNet-50
            - **Params:** 23.5M (20K trainable)
            - **Pretrained:** ImageNet
            - **Fine-tuned:** CIFAR-10
            """)
        else:
            st.markdown("""
            - **Architecture:** EfficientNet-B0
            - **Params:** 4M (12K trainable)
            - **Pretrained:** ImageNet
            - **Fine-tuned:** CIFAR-10
            """)
    
    with col2:
        # Load model and show accuracy
        with st.spinner("Loading victim model..."):
            victim, error = load_victim_model(arch)
        
        if error:
            st.error(error)
            st.info("Run `python victim.py` to train the victim model first.")
            return
        
        st.success(f"✅ Victim model loaded: `{arch}`")
        
        # Compute accuracy
        if st.button("🔍 Compute Test Accuracy", type="primary"):
            with st.spinner("Evaluating on CIFAR-10 test set..."):
                _, test_loader = load_data()
                accuracy = get_victim_accuracy(victim, test_loader, arch)
            
            st.metric("Test Accuracy", f"{accuracy:.2f}%", delta=None)
            
            # Show some predictions
            st.markdown("### Sample Predictions")
            
            cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']
            
            # Get a batch
            images, labels = next(iter(test_loader))
            images = images[:8].to(DEVICE)
            labels = labels[:8]
            
            with torch.no_grad():
                probs = victim.query(images)
                preds = probs.argmax(dim=1).cpu()
            
            # Display
            cols = st.columns(8)
            for i, col in enumerate(cols):
                with col:
                    pred_class = cifar_classes[preds[i]]
                    true_class = cifar_classes[labels[i]]
                    confidence = probs[i, preds[i]].item() * 100
                    
                    if preds[i] == labels[i]:
                        st.markdown(f"✅ **{pred_class}**")
                    else:
                        st.markdown(f"❌ **{pred_class}**")
                    st.caption(f"True: {true_class}")
                    st.caption(f"Conf: {confidence:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Page: Attack Demo
# ══════════════════════════════════════════════════════════════════════════════
def page_attack_demo():
    """Render the attack demonstration page."""
    st.title("⚔️ Attack Demonstration")
    st.markdown("See how model extraction attacks work in real-time.")
    
    st.markdown("---")

    tab_auto, tab_manual = st.tabs(["Automatic Mode", "Manual Mode — query like a real attacker"])

    with tab_auto:
        ensure_owner_defense_state()

        # Configuration
        col1, col2, col3 = st.columns(3)

        with col1:
            attack_name = st.selectbox(
                "Select Attack:",
                ["random_query", "knockoff_nets", "active_learning"],
                format_func=lambda x: {
                    "random_query": "🎲 Random Query",
                    "knockoff_nets": "🔄 Knockoff Nets",
                    "active_learning": "🧠 Active Learning"
                }[x],
            )

        with col2:
            budget = st.select_slider(
                "Query Budget:",
                options=[100, 250, 500, 1000, 2500, 5000],
                value=500,
            )

        with col3:
            sub_arch = st.selectbox(
                "Substitute Architecture:",
                ["small_cnn", "mobilenetv3_small"],
                format_func=lambda x: {
                    "small_cnn": "SmallCNN (95K params)",
                    "mobilenetv3_small": "MobileNetV3 (1.5M params)",
                }[x],
            )

        selected_defense = st.session_state.owner_defense_name
        selected_victim_arch = st.session_state.owner_victim_arch
        st.caption(
            "Server defence is managed centrally in Owner Settings: "
            + ("none (baseline)" if selected_defense == "none" else selected_defense)
        )
        st.caption(f"Active victim architecture (Owner Settings): {selected_victim_arch}")

        # Attack description
        st.markdown("### Attack Strategy")
        descriptions = {
            "random_query": """
            **Random Query Attack** queries the victim with random noise images.
            This is the simplest attack but often ineffective because random noise
            doesn't carry meaningful class information.
            """,
            "knockoff_nets": """
            **Knockoff Nets Attack** queries the victim with real images from the
            same domain (CIFAR-10 test set). The soft probability outputs are used
            as training labels via knowledge distillation (temperature T=3.0).
            """,
            "active_learning": """
            **Active Learning Attack** iteratively selects the most informative
            queries using entropy-based uncertainty sampling. Samples where the
            substitute is most uncertain are prioritized for labeling.
            """,
        }
        st.info(descriptions[attack_name])

        # Run attack
        if st.button("🚀 Launch Attack", type="primary"):
            # Load victim
            victim, error = load_victim_model(selected_victim_arch)
            if error:
                st.error(error)
                return

            _, test_loader = load_data()

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Configure
            config = ThreatConfig(
                budget=budget,
                substitute_arch=sub_arch,
                victim_arch=selected_victim_arch,
            )

            status_text.text("🔄 Running attack...")

            # Time the attack
            start_time = time.time()

            try:
                defense_fn = get_defense_by_name(selected_defense)
                if defense_fn is not None and hasattr(defense_fn, "reset"):
                    defense_fn.reset()

                substitute, meta = run_attack(
                    attack_name,
                    victim,
                    config,
                    defense_fn=defense_fn,
                    sub_epochs=10,  # Reduced for demo
                )

                elapsed = time.time() - start_time
                progress_bar.progress(100)
                status_text.text("✅ Attack complete!")

                # Results
                st.markdown("### Attack Results")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Queries Used", f"{meta.get('queries_used', budget):,}")
                with col2:
                    st.metric("Query Time", f"{meta.get('query_time_s', 0):.1f}s")
                with col3:
                    st.metric("Train Time", f"{meta.get('train_time_s', 0):.1f}s")
                with col4:
                    st.metric("Total Time", f"{elapsed:.1f}s")

                # Compute metrics
                st.markdown("### Extraction Quality")

                with st.spinner("Computing metrics..."):
                    # Load raw model for metrics
                    checkpoint_path = {
                        "resnet50": "models/victim_resnet50.pth",
                        "efficientnet_b0": "models/victim_efficientnet.pth",
                    }.get(selected_victim_arch, f"models/victim_{selected_victim_arch}.pth")

                    victim_model_raw = build_victim_model(selected_victim_arch, pretrained=False)
                    victim_model_raw.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
                    victim_model_raw.eval()

                    victim_baseline_acc = get_victim_accuracy(
                        victim,
                        test_loader,
                        selected_victim_arch,
                    )

                    metrics = compute_all_metrics(
                        victim=victim,
                        victim_model_raw=victim_model_raw,
                        substitute=substitute,
                        test_loader=test_loader,
                        query_budget=budget,
                        defense_fn=defense_fn,
                        victim_acc_baseline=victim_baseline_acc,
                    )

                col1, col2, col3 = st.columns(3)

                with col1:
                    fidelity = metrics['fidelity']
                    st.metric(
                        "Fidelity",
                        f"{fidelity:.1f}%",
                        help="Agreement between victim and substitute"
                    )
                    if fidelity > 70:
                        st.error("⚠️ High extraction success!")
                    elif fidelity > 50:
                        st.warning("⚡ Moderate extraction")
                    else:
                        st.success("✅ Low extraction")

                with col2:
                    sub_acc = metrics['substitute_accuracy']
                    st.metric(
                        "Substitute Accuracy",
                        f"{sub_acc:.1f}%",
                        help="Substitute model's standalone accuracy"
                    )

                with col3:
                    protection = metrics['protection_score']
                    st.metric(
                        "Protection Score",
                        f"{protection:.2f}",
                        help="0=fully extracted, 1=fully protected"
                    )

                st.caption(
                    "Defence: none (baseline)"
                    if selected_defense == "none"
                    else f"Defence: {selected_defense}"
                )

            except Exception as e:
                st.error(f"Attack failed: {e}")
                progress_bar.progress(0)
                status_text.text("❌ Attack failed")

    with tab_manual:
        st.markdown("### Manual Querying")
        st.caption("Query the victim API one image at a time and build attacker data automatically.")

        ensure_owner_defense_state()
        selected_victim_arch = st.session_state.owner_victim_arch
        st.caption(f"Active victim architecture (Owner Settings): {selected_victim_arch}")

        # Session state initialisation
        if "manual_query_images" not in st.session_state:
            st.session_state.manual_query_images = []
        if "manual_query_probs" not in st.session_state:
            st.session_state.manual_query_probs = []
        if "manual_query_log" not in st.session_state:
            st.session_state.manual_query_log = []
        if "manual_last_output" not in st.session_state:
            st.session_state.manual_last_output = None
        if "manual_last_pred_idx" not in st.session_state:
            st.session_state.manual_last_pred_idx = None
        if "manual_last_trained_count" not in st.session_state:
            st.session_state.manual_last_trained_count = 0
        if "manual_substitute" not in st.session_state:
            st.session_state.manual_substitute = None
        if "manual_latest_fidelity" not in st.session_state:
            st.session_state.manual_latest_fidelity = None

        # Controls
        c1, c2 = st.columns(2)

        with c1:
            source_mode = st.radio(
                "Image Source",
                ["CIFAR-10 test index", "Upload image"],
                horizontal=True,
            )

        with c2:
            manual_sub_arch = st.selectbox(
                "Substitute Architecture (auto-train)",
                ["small_cnn", "mobilenetv3_small"],
                format_func=lambda x: {
                    "small_cnn": "SmallCNN (95K params)",
                    "mobilenetv3_small": "MobileNetV3 (1.5M params)",
                }[x],
                key="manual_sub_arch",
            )

        query_image_tensor = None
        query_source_text = ""

        _, test_loader = load_data()
        test_dataset = test_loader.dataset

        if source_mode == "CIFAR-10 test index":
            max_idx = len(test_dataset) - 1
            idx = st.number_input(
                "Test image index",
                min_value=0,
                max_value=max_idx,
                value=0,
                step=1,
            )
            image_tensor, _ = test_dataset[int(idx)]
            query_image_tensor = image_tensor.unsqueeze(0)
            query_source_text = f"cifar_test[{int(idx)}]"
        else:
            uploaded = st.file_uploader(
                "Upload an image",
                type=["png", "jpg", "jpeg"],
                key="manual_upload",
            )
            if uploaded is not None:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Uploaded image", width=180)
                upload_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
                ])
                query_image_tensor = upload_transform(image).unsqueeze(0)
                query_source_text = uploaded.name

        b1, b2 = st.columns([2, 1])
        send_query = b1.button("📨 Send Query to Victim API", type="primary")
        clear_all = b2.button("🧹 Clear and start over")

        if clear_all:
            st.session_state.manual_query_images = []
            st.session_state.manual_query_probs = []
            st.session_state.manual_query_log = []
            st.session_state.manual_last_output = None
            st.session_state.manual_last_pred_idx = None
            st.session_state.manual_last_trained_count = 0
            st.session_state.manual_substitute = None
            st.session_state.manual_latest_fidelity = None
            if st.session_state.owner_defense_fn is not None and hasattr(st.session_state.owner_defense_fn, "reset"):
                st.session_state.owner_defense_fn.reset()
            st.success("Cleared. Start sending new queries.")

        if send_query:
            victim, error = load_victim_model(selected_victim_arch)
            if error:
                st.error(error)
                return

            if query_image_tensor is None:
                st.warning("Select a valid image source first.")
            else:
                with torch.no_grad():
                    probs = victim.query(query_image_tensor)

                defense_obj = st.session_state.owner_defense_fn
                if defense_obj is not None:
                    probs = defense_obj(probs)

                pred_idx = int(probs.argmax(dim=1).item())
                confidence = float(probs[0, pred_idx].item() * 100.0)
                query_no = len(st.session_state.manual_query_images) + 1

                # Store attacker data automatically
                st.session_state.manual_query_images.append(query_image_tensor.cpu())
                st.session_state.manual_query_probs.append(probs.cpu())
                st.session_state.manual_last_output = probs.squeeze(0).cpu().numpy()
                st.session_state.manual_last_pred_idx = pred_idx

                st.session_state.manual_query_log.append({
                    "query_no": query_no,
                    "source": query_source_text,
                    "returned_class": CIFAR10_CLASSES[pred_idx],
                    "confidence_%": round(confidence, 2),
                })

                st.success("Query sent and stored for substitute training.")

        # Show latest probability vector
        if st.session_state.manual_last_output is not None:
            st.markdown("### Latest Victim API Output")
            probs_arr = st.session_state.manual_last_output
            pred_idx = st.session_state.manual_last_pred_idx

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(CIFAR10_CLASSES),
                y=probs_arr,
                marker_color="#3498db",
            ))
            fig.update_layout(
                title="Returned probability vector (10 classes)",
                xaxis_title="Class",
                yaxis_title="Probability",
                yaxis=dict(range=[0, 1]),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.metric(
                "Attacker best guess",
                str(CIFAR10_CLASSES[pred_idx]),
                f"{probs_arr[pred_idx] * 100:.2f}% confidence",
            )

        # Auto-train substitute after 20+ queries
        query_count = len(st.session_state.manual_query_images)
        st.markdown(f"### Collected Query Pairs: {query_count}")

        if query_count >= 20 and query_count != st.session_state.manual_last_trained_count:
            victim, error = load_victim_model(selected_victim_arch)
            if error:
                st.error(error)
                return

            _, manual_test_loader = load_data()

            with st.spinner("Auto-training substitute (triggered at 20+ collected queries)..."):
                substitute = build_substitute_model(manual_sub_arch)
                substitute = train_substitute(
                    substitute,
                    st.session_state.manual_query_images,
                    st.session_state.manual_query_probs,
                    epochs=8,
                    batch_size=32,
                )

                # Fidelity against victim outputs
                total = 0
                agree = 0
                eval_defense = get_defense_by_name(st.session_state.owner_defense_name)
                if eval_defense is not None and hasattr(eval_defense, "reset"):
                    eval_defense.reset()

                with torch.no_grad():
                    for eval_imgs, _ in manual_test_loader:
                        victim_probs = victim.query(eval_imgs)
                        if eval_defense is not None:
                            victim_probs = eval_defense(victim_probs)
                        victim_preds = victim_probs.argmax(dim=1)

                        logits = substitute(eval_imgs.to(DEVICE, non_blocking=True))
                        sub_preds = logits.argmax(dim=1).cpu()

                        agree += (victim_preds == sub_preds).sum().item()
                        total += eval_imgs.size(0)

                fidelity = 100.0 * agree / total if total > 0 else 0.0
                st.session_state.manual_latest_fidelity = fidelity
                st.session_state.manual_substitute = substitute
                st.session_state.manual_last_trained_count = query_count

        if st.session_state.manual_latest_fidelity is not None:
            st.markdown("### Auto-training Result")
            st.metric(
                "Substitute fidelity",
                f"{st.session_state.manual_latest_fidelity:.2f}%",
                help="Agreement between victim API outputs and substitute predictions",
            )

        # Query log table
        st.markdown("### Query Log")
        if len(st.session_state.manual_query_log) == 0:
            st.caption("No queries yet.")
        else:
            log_df = pd.DataFrame(st.session_state.manual_query_log)
            visible_cols = ["query_no", "source", "returned_class", "confidence_%"]
            st.dataframe(log_df[visible_cols], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page: Owner Settings
# ══════════════════════════════════════════════════════════════════════════════
def page_owner_settings():
    """Render the owner/server-side settings page."""
    st.title("🔒 Owner Settings")
    st.markdown("Configure server-side behaviour hidden from attackers.")
    st.markdown("---")

    ensure_owner_defense_state()

    st.info(
        "This setting represents the defence configured by the company on the server. "
        "Attackers do not see this page and do not know which defence is active."
    )

    defenses = get_all_defenses()
    defense_names = [name for name, _ in defenses]
    defense_options = ["none"] + [d for d in defense_names if d != "none"]

    victim_arch_options = ["resnet50"]
    current_victim_arch = st.session_state.owner_victim_arch
    current_victim_idx = (
        victim_arch_options.index(current_victim_arch)
        if current_victim_arch in victim_arch_options else 0
    )

    selected_owner_victim_arch = st.selectbox(
        "Victim Architecture",
        victim_arch_options,
        index=current_victim_idx,
        format_func=lambda x: "ResNet-50" if x == "resnet50" else "EfficientNet-B0",
        key="owner_victim_arch_select",
    )
    if selected_owner_victim_arch != st.session_state.owner_victim_arch:
        st.session_state.owner_victim_arch = selected_owner_victim_arch

    current = st.session_state.owner_defense_name
    current_idx = defense_options.index(current) if current in defense_options else 0

    selected_owner_defense = st.selectbox(
        "Server Defence",
        defense_options,
        index=current_idx,
        format_func=lambda x: "None" if x == "none" else x,
        key="owner_defense_select",
    )

    if selected_owner_defense != st.session_state.owner_defense_name:
        st.session_state.owner_defense_name = selected_owner_defense
        st.session_state.owner_defense_fn = get_defense_by_name(selected_owner_defense)
        if st.session_state.owner_defense_fn is not None and hasattr(st.session_state.owner_defense_fn, "reset"):
            st.session_state.owner_defense_fn.reset()

    st.success(
        "Active server defence: none (baseline)"
        if st.session_state.owner_defense_name == "none"
        else f"Active server defence: {st.session_state.owner_defense_name}"
    )
    st.success(f"Active victim architecture: {st.session_state.owner_victim_arch}")


# ══════════════════════════════════════════════════════════════════════════════
# Page: Defense Demo
# ══════════════════════════════════════════════════════════════════════════════
def page_defense_demo():
    """Render the defense demonstration page."""
    st.title("🛡️ Defense Demonstration")
    st.markdown("See how defenses modify model outputs to hinder extraction.")
    
    st.markdown("---")
    
    # Defense selection
    defenses = get_all_defenses()
    defense_names = [name for name, _ in defenses]
    
    selected_defense = st.selectbox(
        "Select Defense:",
        defense_names,
        format_func=lambda x: x.replace("_", " ").title(),
    )
    
    defense_fn = get_defense_by_name(selected_defense)
    
    # Defense explanation
    st.markdown("### How This Defense Works")
    
    explanations = {
        "none":                 "No defense applied — raw probability outputs are returned to the attacker.",
        "throttle_250":         "🟢 STRONG — Allow only 250 total queries; after that return uniform noise. Fidelity drop ~49 pp.",
        "throttle_500":         "🟢 STRONG — Allow only 500 total queries; after that return uniform noise. Fidelity drop ~49 pp.",
        "throttle_1000":        "🟢 STRONG — Allow only 1000 total queries; after that return uniform noise. Fidelity drop ~40 pp.",
        "poison_0.30":          "🟡 MODERATE — Randomly shuffle 30% of returned probability vectors. Fidelity drop ~16 pp.",
        "poison_0.50":          "🟡 MODERATE — Randomly shuffle 50% of returned probability vectors. Fidelity drop ~26 pp.",
        "adaptive_aggressive":  "🟡 MODERATE — Escalate noise 10× when query rate exceeds 30 q/s (suspicious pattern). Fidelity drop ~7 pp.",
    }
    
    st.info(explanations.get(selected_defense, "Defense mechanism"))
    
    # Visual demonstration
    st.markdown("### Visual Demonstration")
    
    # Create sample probability vector
    np.random.seed(42)
    original_probs = np.array([0.05, 0.02, 0.01, 0.82, 0.03, 0.02, 0.01, 0.02, 0.01, 0.01])
    original_probs = original_probs / original_probs.sum()  # Normalize
    
    # Apply defense
    if defense_fn is not None:
        probs_tensor = torch.tensor(original_probs, dtype=torch.float32).unsqueeze(0)
        defended_tensor = defense_fn(probs_tensor)
        defended_probs = defended_tensor.squeeze(0).numpy()
    else:
        defended_probs = original_probs
    
    # Plot comparison
    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                     'dog', 'frog', 'horse', 'ship', 'truck']
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Original Output", "After Defense"])
    
    fig.add_trace(
        go.Bar(x=cifar_classes, y=original_probs, marker_color='#e74c3c', name="Original"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=cifar_classes, y=defended_probs, marker_color='#2ecc71', name="Defended"),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_yaxes(range=[0, 1])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Numerical comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Probabilities:**")
        orig_df = pd.DataFrame({
            "Class": cifar_classes,
            "Probability": [f"{p:.4f}" for p in original_probs],
        })
        st.dataframe(orig_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Defended Probabilities:**")
        def_df = pd.DataFrame({
            "Class": cifar_classes,
            "Probability": [f"{p:.4f}" for p in defended_probs],
        })
        st.dataframe(def_df, use_container_width=True, hide_index=True)
    
    # Impact analysis
    st.markdown("### Defense Impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_changed = np.argmax(original_probs) != np.argmax(defended_probs)
        if pred_changed:
            st.error("❌ Prediction Changed!")
        else:
            st.success("✅ Prediction Preserved")
    
    with col2:
        info_loss = np.sum(np.abs(original_probs - defended_probs))
        st.metric("Information Distortion", f"{info_loss:.3f}")
    
    with col3:
        entropy_orig = -np.sum(original_probs * np.log(original_probs + 1e-10))
        entropy_def = -np.sum(defended_probs * np.log(defended_probs + 1e-10))
        entropy_change = entropy_def - entropy_orig
        st.metric("Entropy Change", f"{entropy_change:+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Page: Results Analysis
# ══════════════════════════════════════════════════════════════════════════════
def page_results_analysis():
    """Render the results analysis page."""
    st.title("📊 Results Analysis")
    st.markdown("Analyze experiment results from CSV files.")
    
    st.markdown("---")
    
    # Check for results files
    results_dir = "experiments/results"
    
    if not os.path.exists(results_dir):
        st.warning("No results directory found. Run `python evaluate.py` first.")
        return
    
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    if not csv_files:
        st.warning("No CSV result files found. Run `python evaluate.py` first.")
        
        # Show sample data structure
        st.markdown("### Expected Data Structure")
        sample_df = pd.DataFrame({
            "attack": ["knockoff_nets", "knockoff_nets"],
            "defense": ["none", "throttle_500"],
            "budget": [5000, 5000],
            "substitute_arch": ["mobilenetv3_small", "mobilenetv3_small"],
            "fidelity": [70.1, 21.5],
            "substitute_accuracy": [67.9, 20.0],
            "protection_score": [0.03, 0.71],
        })
        st.dataframe(sample_df, use_container_width=True)
        return
    
    # File selection
    selected_file = st.selectbox("Select Results File:", csv_files)
    
    # Load data
    df = pd.read_csv(os.path.join(results_dir, selected_file))
    
    st.success(f"✅ Loaded {len(df)} experiment results")
    
    # Filters
    st.markdown("### Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        attacks = st.multiselect(
            "Attacks:",
            df['attack'].dropna().unique().tolist(),
            default=df['attack'].dropna().unique().tolist(),
        )
    
    with col2:
        if 'defense' in df.columns:
            defenses = st.multiselect(
                "Defenses:",
                df['defense'].dropna().unique().tolist(),
                default=df['defense'].dropna().unique().tolist(),
            )
        else:
            defenses = []
    
    with col3:
        budgets = st.multiselect(
            "Budgets:",
            df['budget'].dropna().unique().tolist(),
            default=df['budget'].dropna().unique().tolist(),
        )
    
    # Filter dataframe
    filtered_df = df[
        df['attack'].isin(attacks) &
        df['budget'].isin(budgets)
    ]
    if defenses and 'defense' in df.columns:
        filtered_df = filtered_df[filtered_df['defense'].isin(defenses)]
    
    st.markdown(f"**Showing {len(filtered_df)} results**")
    
    # Summary statistics
    st.markdown("### Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_fidelity = filtered_df['fidelity'].mean()
        st.metric("Avg Fidelity", f"{avg_fidelity:.1f}%")
    
    with col2:
        max_fidelity = filtered_df['fidelity'].max()
        st.metric("Max Fidelity", f"{max_fidelity:.1f}%")
    
    with col3:
        if 'protection_score' in filtered_df.columns:
            avg_protection = filtered_df['protection_score'].mean()
            st.metric("Avg Protection", f"{avg_protection:.2f}")
    
    with col4:
        if 'substitute_accuracy' in filtered_df.columns:
            max_acc = filtered_df['substitute_accuracy'].max()
            st.metric("Max Sub Accuracy", f"{max_acc:.1f}%")
    
    # ── Defence Effectiveness Scorecard ─────────────────────────────────────
    if 'defense' in filtered_df.columns and 'fidelity' in filtered_df.columns:
        st.markdown("### 🛡️ Defence Effectiveness Scorecard")

        # Determine baseline fidelity from the "none" row in the full (unfiltered) df
        baseline_rows = df[df['defense'] == 'none'] if 'defense' in df.columns else pd.DataFrame()
        if len(baseline_rows) > 0:
            baseline_fidelity = baseline_rows['fidelity'].mean()
        else:
            # Fall back to the max fidelity in the current filtered set
            baseline_fidelity = filtered_df['fidelity'].max()

        st.caption(
            f"📌 Baseline (no defence) fidelity: **{baseline_fidelity:.1f}%** — "
            "a good defence pushes attacker fidelity well below this value."
        )

        # Build enriched display dataframe
        display_df = filtered_df.copy()
        display_df['Fidelity Drop (pp)'] = (baseline_fidelity - display_df['fidelity']).round(2)

        def _eff_badge(drop: float) -> str:
            if drop >= 30:
                return "🟢 STRONG"
            elif drop >= 10:
                return "🟡 MODERATE"
            elif drop >= 0:
                return "🔴 WEAK"
            else:
                return "⚪ NO EFFECT (worse than none)"

        display_df['Effectiveness'] = display_df['Fidelity Drop (pp)'].apply(_eff_badge)

        # Quick count summary cards
        n_strong   = int((display_df['Fidelity Drop (pp)'] >= 30).sum())
        n_moderate = int(((display_df['Fidelity Drop (pp)'] >= 10) & (display_df['Fidelity Drop (pp)'] < 30)).sum())
        n_weak     = int((display_df['Fidelity Drop (pp)'] < 10).sum())

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("🟢 Strong defenders",   str(n_strong),
                   help="Fidelity drop ≥ 30 pp — attacker fails badly")
        sc2.metric("🟡 Moderate defenders", str(n_moderate),
                   help="Fidelity drop 10–30 pp — partial protection")
        sc3.metric("🔴 Weak / no effect",   str(n_weak),
                   help="Fidelity drop < 10 pp — defence barely helps")

        st.markdown("---")

        # Shared column order for sub-tables
        _show_cols = [c for c in [
            'defense', 'attack', 'budget',
            'fidelity', 'substitute_accuracy',
            'Fidelity Drop (pp)', 'Effectiveness',
        ] if c in display_df.columns]

        # 🟢 Strong defenders
        strong_df = (display_df[display_df['Fidelity Drop (pp)'] >= 30]
                     .sort_values('Fidelity Drop (pp)', ascending=False)
                     .reset_index(drop=True))
        if len(strong_df) > 0:
            st.markdown("#### 🟢 Strong Defenders — effectively block model stealing")
            st.dataframe(strong_df[_show_cols], use_container_width=True)

        # 🟡 Moderate defenders
        mod_df = (display_df[
            (display_df['Fidelity Drop (pp)'] >= 10) &
            (display_df['Fidelity Drop (pp)'] < 30)
        ].sort_values('Fidelity Drop (pp)', ascending=False)
         .reset_index(drop=True))
        if len(mod_df) > 0:
            st.markdown("#### 🟡 Moderate Defenders — some protection, but attacker still partially succeeds")
            st.dataframe(mod_df[_show_cols], use_container_width=True)

        # 🔴 Weak / no-effect
        weak_df = (display_df[display_df['Fidelity Drop (pp)'] < 10]
                   .sort_values('Fidelity Drop (pp)', ascending=False)
                   .reset_index(drop=True))
        if len(weak_df) > 0:
            st.markdown("#### 🔴 Weak / No-Effect Defenders — attacker still succeeds")
            st.dataframe(weak_df[_show_cols], use_container_width=True)

        st.markdown("---")

        # Full table with badge column
        st.markdown("### 📋 Full Data Table (with Effectiveness label)")
        st.dataframe(display_df, use_container_width=True)

    else:
        # Fallback: no defense column — plain table
        st.markdown("### Data Table")
        st.dataframe(filtered_df, use_container_width=True)

    # Download
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        "📥 Download Filtered Results",
        csv_data,
        file_name="filtered_results.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Page: Visualizations
# ══════════════════════════════════════════════════════════════════════════════
def page_visualizations():
    """Render the visualizations page."""
    st.title("📈 Visualizations")
    st.markdown("Interactive charts for exploring results.")
    
    st.markdown("---")
    
    # Check for results
    results_dir = "experiments/results"
    csv_files = []
    if os.path.exists(results_dir):
        csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    if not csv_files:
        # Demo with sample data
        st.info("No results files found. Showing demo visualizations.")
        
        # Sample data
        df = pd.DataFrame({
            "attack": ["random_query"]*6 + ["knockoff_nets"]*6 + ["active_learning"]*6,
            "budget": [1000, 5000, 10000]*6,
            "substitute_arch": ["small_cnn", "small_cnn", "small_cnn", 
                               "mobilenetv3_small", "mobilenetv3_small", "mobilenetv3_small"]*3,
            "fidelity": [10, 10, 10, 10, 10, 10,  # random
                        39, 51, 55, 10, 75, 84,  # knockoff
                        37, 51, 55, 9, 77, 85],  # active
            "substitute_accuracy": [10, 10, 10, 10, 10, 10,
                                   38, 49, 53, 10, 71, 81,
                                   35, 48, 53, 10, 73, 82],
            "protection_score": [0.88, 0.88, 0.88, 0.88, 0.88, 0.88,
                                0.54, 0.41, 0.36, 0.88, 0.14, 0.02,
                                0.58, 0.41, 0.36, 0.88, 0.12, 0.01],
        })
    else:
        selected_file = st.selectbox("Select Results File:", csv_files)
        df = pd.read_csv(os.path.join(results_dir, selected_file))
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Fidelity by Attack",
        "📈 Budget vs Fidelity",
        "🎯 Attack Comparison",
        "🛡️ Defense Heatmap",
    ])
    
    with tab1:
        st.markdown("### Fidelity by Attack Strategy")
        
        fig = px.box(
            df, x="attack", y="fidelity",
            color="attack",
            title="Distribution of Fidelity Scores by Attack",
            labels={"fidelity": "Fidelity (%)", "attack": "Attack Strategy"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Query Budget vs Extraction Success")
        
        fig = px.line(
            df.groupby(['attack', 'budget'])['fidelity'].mean().reset_index(),
            x="budget", y="fidelity",
            color="attack",
            markers=True,
            title="Fidelity vs Query Budget",
            labels={"fidelity": "Fidelity (%)", "budget": "Query Budget"},
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Attack Strategy Comparison")
        
        comparison_df = df.groupby('attack').agg({
            'fidelity': 'mean',
            'substitute_accuracy': 'mean',
            'protection_score': 'mean',
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Fidelity',
            x=comparison_df['attack'],
            y=comparison_df['fidelity'],
            marker_color='#e74c3c',
        ))
        
        fig.add_trace(go.Bar(
            name='Substitute Acc',
            x=comparison_df['attack'],
            y=comparison_df['substitute_accuracy'],
            marker_color='#3498db',
        ))
        
        fig.update_layout(
            title="Average Metrics by Attack",
            barmode='group',
            yaxis_title="Percentage (%)",
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Defense Effectiveness Heatmap")
        
        if 'defense' in df.columns and df['defense'].nunique() > 1:
            pivot_df = df.pivot_table(
                values='fidelity',
                index='attack',
                columns='defense',
                aggfunc='mean',
            )
            
            fig = px.imshow(
                pivot_df,
                labels=dict(x="Defense", y="Attack", color="Fidelity (%)"),
                title="Fidelity by Attack-Defense Combination",
                color_continuous_scale="RdYlGn_r",  # Red=high fidelity (bad), Green=low (good)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Defense heatmap requires results with multiple defenses (Phase 3).")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    """Main entry point."""
    if not IMPORTS_OK:
        st.error(f"Import Error: {IMPORT_ERROR}")
        st.info("Make sure you're running from the project directory.")
        return
    
    page = render_sidebar()
    
    if page == "🏠 Overview":
        page_overview()
    elif page == "🎯 Victim Model":
        page_victim_model()
    elif page == "⚔️ Attack Demo":
        page_attack_demo()
    elif page == "🔒 Owner Settings":
        page_owner_settings()
    elif page == "🛡️ Defense Demo":
        page_defense_demo()
    elif page == "📊 Results Analysis":
        page_results_analysis()
    elif page == "📈 Visualizations":
        page_visualizations()


if __name__ == "__main__":
    main()
