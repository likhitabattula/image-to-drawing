function convertImage() {
  const input = document.getElementById('imageInput');
  const file = input.files[0];
  const effect = document.getElementById('effect').value;

  const formData = new FormData();
  formData.append('image', file);
  formData.append('effect', effect); // Add selected effect

  fetch('http://127.0.0.1:5000/convert', {
    method: 'POST',
    body: formData
  })
    .then(response => response.blob())
    .then(blob => {
      const imageURL = URL.createObjectURL(blob);
      document.getElementById('result').src = imageURL;

      // Enable download button
      const downloadLink = document.getElementById('downloadLink');
      downloadLink.href = imageURL;
      downloadLink.style.display = 'inline';
    })
    .catch(error => console.error('Error:', error));
}
