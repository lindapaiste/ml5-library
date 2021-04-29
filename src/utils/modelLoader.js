function isAbsoluteURL(str) {
  const pattern = new RegExp('^(?:[a-z]+:)?//', 'i');
  return !!pattern.test(str);
}

function getModelPath(absoluteOrRelativeUrl) {
  return isAbsoluteURL(absoluteOrRelativeUrl) ? absoluteOrRelativeUrl : window.location.pathname + absoluteOrRelativeUrl;
}

export default {
  isAbsoluteURL,
  getModelPath
}