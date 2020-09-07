mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = True\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
