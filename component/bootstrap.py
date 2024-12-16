import streamlit as st

def bootstrap():
    css = """
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" integrity="sha384-pzjw8f+ua7Kw1TIqjQvF0MpmZ4gy63E5ghK9ZkNfjLC/jKksNStlXKp4YfRvH+8A" crossorigin="anonymous">
    """
    js = """
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js" integrity="sha384-pzjw8f+ua7Kw1TIqjQvF0MpmZ4gy63E5ghK9ZkNfjLC/jKksNStlXKp4YfRvH+8A" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js" integrity="sha384-B4gt1jrGC7Jh4Ag+CZfnwFlJ9x60UpLS6CjG3uH9S2Ih5VQ5sl5QwLw3jRz0iwc5" crossorigin="anonymous"></script>
    """
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(js, unsafe_allow_html=True)