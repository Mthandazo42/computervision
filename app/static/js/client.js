var el = x => document.getElementById(x);

function analyze() {
  var uploadFiles = el("inp_file").files;
  if (uploadFiles.length !== 1) alert("Please select a file to analyze!");

  el("submit_btn").innerHTML = "Analyzing...";
  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`,
    true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.onload = function(e) {
    if (this.readyState === 4) {
      var response = JSON.parse(e.target.responseText);
      el("result-label").innerHTML = `The breed type of your pet is ${response["result"]}`;
    }
    el("submit_btn").innerHTML = "Analyze";
  };

  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);
  xhr.send(fileData);
}