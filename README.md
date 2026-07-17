# TM2-RAG: Temporal Multimodal Retrieval-Augmented Generation

Welcome to the TM2-RAG repository. This project implements a comprehensive, multimodal, and time-aware retrieval-augmented generation pipeline tailored for longitudinal clinical data (such as chest X-rays and their corresponding radiology reports). 

This guide will walk you through the end-to-end process of running the full TM2-RAG system pipeline, starting from raw data ingestion all the way to interactive inference via our Streamlit application.

## Prerequisites

Before running the pipeline, ensure you have the required datasets available (e.g., MIMIC-CXR) and all necessary Python dependencies installed in your environment. You should configure the data paths in `src/config.py` appropriately to match your local setup.

## Pipeline Execution Steps

The system processes data in a sequence of defined phases. Run the following Python scripts in order from the root of the workspace.

### Phase 1: Data Indexing and Filtering
First, we build a master index linking patients, studies, images, and reports, and then filter it down to patients that possess longitudinal (temporal) data.

1. **Phase 1A: Build Master Index**
   Run the master index builder to map out the entire dataset structure.
   ```bash
   python src/data/build_master_index.py
   ```

2. **Phase 1B: Filter Temporal Patients**
   Filter the master index to retain only the patients with multiple studies over time.
   ```bash
   python src/data/filter_temporal_patients.py
   ```

### Phase 2: NLP and Vision Feature Extraction
In this phase, we extract structured sections from the radiology reports, detect temporal language, and process visual features.

3. **Phase 2 (2A, 2B, 2C): Extract All Features**
   This orchestrator script runs report parsing (2A) and temporal change detection (2B) to extract comprehensive NLP features from the reports.
   ```bash
   python src/nlp/extract_all_features.py
   ```

### Phase 3: Temporal Trajectory Construction
Next, we reconstruct the longitudinal patient timelines by building trajectories, analyzing transitions, and defining dynamic windows.

4. **Phase 3A: Trajectory Builder**
   Construct temporal patient trajectories from their historical studies.
   ```bash
   python src/temporal/trajectory_builder.py
   ```

5. **Phase 3B: Transition Analyzer**
   Analyze the transitions and changes between consecutive studies.
   ```bash
   python src/temporal/transition_analyzer.py
   ```

6. **Phase 3C: Dynamic Windowing**
   Create dynamic temporal windows for each patient's trajectory to optimize context retrieval.
   ```bash
   python src/temporal/dynamic_window.py
   ```

### Phase 4: Dense Embeddings Generation
We now embed the reports and temporal trajectories into a dense vector space for semantic retrieval.

7. **Phase 4A: Report Embeddings**
   Generate embeddings for the individual reports.
   ```bash
   python src/embeddings/report_embedder.py
   ```

8. **Phase 4B: Trajectory and Window Embeddings**
   Generate embeddings that capture the longitudinal temporal trajectories and dynamic windows.
   ```bash
   python src/embeddings/trajectory_embedder.py
   ```

### Phase 5 & 6: Image Embeddings and Vector Store Compilation
We process visual inputs (specifically handling multiple views like AP and LV) and finally compile our embeddings into vector databases.

9. **Phase 5: Dual Image Embeddings**
   Extract offline features for the dual-view chest X-rays.
   ```bash
   python src/embeddings/dual_image_embedder.py
   ```

10. **Phase 6: Build Vector Stores**
    Compile all the extracted embeddings into efficient vector stores for fast similarity search.
    ```bash
    python src/embeddings/vector_store.py
    ```

### Phase 7: Multimodal Fusion Index
11. **Phase 7: Fused Vector Store (View-Aware)**
    Fuse the text and image embeddings to build the ultimate cross-modal, view-aware fused vector store using BiomedCLIP.
    ```bash
    python src/fusion/build_fusion_index.py
    ```

## Running the TM2-RAG Application

Once the entire offline data pipeline is executed and all indices and vector stores are built, you can run the interactive Chatbot interface. The application relies on `Phase 5A` (Query Processor) and `Phase 5B` (Time-Aware Retriever) at runtime to fetch the most relevant context for generation.

To start the Streamlit application:
```bash
streamlit run app.py
```

This will launch a web interface where you can input real-time text and multimodal chest X-rays, query the system, perform longitudinal trajectory analysis, and generate explainable clinical findings.
