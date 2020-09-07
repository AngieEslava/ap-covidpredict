mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
CORSenabled = True\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
