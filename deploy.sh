#!/bin/bash
# Deployment script for Trade Network Intelligence
# Usage: ./deploy.sh [local|docker|cloud]

set -e

MODE=${1:-local}
DATA_DIR="$HOME/trade_data_warehouse"

case $MODE in
    local)
        echo "Starting local server..."
        streamlit run app.py --server.port 8501
        ;;

    docker)
        echo "Building Docker image..."
        # Copy data into build context (exclude large HS product files for smaller image)
        mkdir -p data/baci data/gravity data/tau
        cp "$DATA_DIR/baci/baci_bilateral_totals.parquet" data/baci/
        cp "$DATA_DIR/baci/country_codes.parquet" data/baci/
        cp "$DATA_DIR/baci/bilateral_sector_flows.parquet" data/baci/
        cp "$DATA_DIR/baci/product_codes_hs02.parquet" data/baci/
        cp "$DATA_DIR/gravity/gravity_v202211.parquet" data/gravity/
        cp -r "$DATA_DIR/tau/" data/tau/
        # Optionally include HS product data (adds ~1.6GB)
        # cp -r "$DATA_DIR/baci/hs_by_dyad/" data/baci/hs_by_dyad/

        docker build -t trade-network-viz .
        echo "Running container..."
        docker run -p 8501:8501 trade-network-viz
        ;;

    cloud)
        echo "=== Cloud Deployment Options ==="
        echo ""
        echo "1. STREAMLIT COMMUNITY CLOUD (Free, easiest)"
        echo "   - Push to GitHub, connect at share.streamlit.io"
        echo "   - Limit: 1GB, so exclude hs_by_dyad/ data"
        echo "   - Add data to .gitignore, use git-lfs or external storage"
        echo ""
        echo "2. RAILWAY.APP (Free tier available)"
        echo "   - railway login && railway init && railway up"
        echo "   - Supports larger data, Docker-based"
        echo ""
        echo "3. GOOGLE CLOUD RUN (Pay-per-use, very cheap)"
        echo "   - gcloud run deploy trade-viz --source . --region us-central1"
        echo "   - Mount data from GCS bucket"
        echo ""
        echo "4. DIGITAL OCEAN APP PLATFORM (\$5/mo)"
        echo "   - doctl apps create --spec app.yaml"
        echo ""
        echo "5. RENDER.COM (Free tier)"
        echo "   - Connect GitHub repo, auto-deploys"
        echo ""
        echo "For any cloud option, you'll need to:"
        echo "  a) Push code to GitHub"
        echo "  b) Handle data: either bundle the ~93MB core data, or"
        echo "     store in cloud storage (GCS/S3) and load at runtime"
        ;;

    *)
        echo "Usage: ./deploy.sh [local|docker|cloud]"
        exit 1
        ;;
esac
