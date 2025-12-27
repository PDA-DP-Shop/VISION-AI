def audio_page():
    return render_template('audio.html')

@app.route('/text')
def text_page():
    return render_template('text.html')

if __name__ == '__main__':
    app.run(port=5001, debug=True)
