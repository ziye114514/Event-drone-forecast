// Local adapter only: invokes the user's installed Metavision SDK headers.
// No SDK source is copied or redistributed here. Requires the SDK's license.
#ifndef __AVX2__
#error AVX2_compiler_target_is_required_for_this_private_adapter
#endif
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/cv/algorithms/activity_noise_filter_algorithm.h>
#include <vector>
#include <iterator>
#include <cstddef>
namespace py=pybind11;

class ActivityAdapter {
    Metavision::ActivityNoiseFilterAlgorithm<> filter_;
    std::vector<Metavision::EventCD> output_;
public:
    ActivityAdapter(unsigned width,unsigned height,long long threshold):filter_(width,height,threshold) {
        output_.resize(2000000);output_.clear();
    }
    py::array process(py::array events) {
        auto info=events.request();
        if(info.ndim!=1 || info.itemsize!=sizeof(Metavision::EventCD) || (info.shape[0]>0 && info.strides[0]!=sizeof(Metavision::EventCD)))
            throw py::value_error("Expected contiguous native SDK EventCD records");
        auto fields=events.dtype().attr("fields").cast<py::dict>();
        const char* names[]={"x","y","p","t"};
        const size_t offsets[]={offsetof(Metavision::EventCD,x),offsetof(Metavision::EventCD,y),offsetof(Metavision::EventCD,p),offsetof(Metavision::EventCD,t)};
        for(int i=0;i<4;i++) {
            if(!fields.contains(names[i]) || fields[py::str(names[i])].cast<py::tuple>()[1].cast<size_t>()!=offsets[i])
                throw py::value_error("Native SDK EventCD field offsets differ");
        }
        auto first=static_cast<const Metavision::EventCD*>(info.ptr);
        {
            py::gil_scoped_release release;
            output_.clear();output_.reserve(info.shape[0]);
            filter_.process_events(first,first+info.shape[0],std::back_inserter(output_));
        }
        return py::array(events.dtype(),{static_cast<py::ssize_t>(output_.size())},
            {static_cast<py::ssize_t>(sizeof(Metavision::EventCD))},output_.data(),py::cast(this));
    }
};
PYBIND11_MODULE(_edf_sdk_activity,m) {
    m.attr("uses_installed_SDK_header")=true;m.attr("compiled_AVX2")=true;
    py::class_<ActivityAdapter>(m,"ActivityAdapter").def(py::init<unsigned,unsigned,long long>()).def("process",&ActivityAdapter::process);
}
