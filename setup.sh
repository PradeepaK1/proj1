mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"ptljkpd@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = 8502\n\
" > ~/.streamlit/config.toml
