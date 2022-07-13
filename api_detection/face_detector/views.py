# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
import urllib.request as urllib
from .services.detector import haarcascade_frontalface_default, dlibDetector, convertFromBytes

@csrf_exempt
def detect(request):
    response = {"success": False}
    if request.method == "POST":
        if request.FILES.get("image", None) is not None:
                data_image = _grab_image(stream=request.FILES.get("image"))
        else:
            url = request.POST.get("url", None)
            if url is None:
                response["error"] = "No URL provided."
                return JsonResponse(response)
            data_image = _grab_image(url=url)
        
        mod = request.POST.get('algorithm', 'dlib')
        data_image = convertFromBytes(data_image)
        if mod == 'dlib':
            rects = dlibDetector(data_image)
        else:
            rects = haarcascade_frontalface_default(data_image)
        response.update({"num_faces": len(rects), "faces": rects, "success": True})
    return JsonResponse(response)


def _grab_image(stream=None, url=None):
    if url is not None:
        resp = urllib.urlopen(url)
        data = resp.read()
    elif stream is not None:
        data = stream.read()
    return data
