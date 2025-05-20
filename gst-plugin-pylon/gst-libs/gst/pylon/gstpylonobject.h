/* Copyright (C) 2022 Basler AG
 *
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *     1. Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *     2. Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *     3. Neither the name of the copyright holder nor the names of
 *        its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written
 *        permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _GST_PYLON_OBJECT_H_
#define _GST_PYLON_OBJECT_H_

#include <gst/gst.h>
#include <gst/pylon/gstpyloncache.h>
#include <gst/pylon/gstpylonincludes.h>

G_DECLARE_DERIVABLE_TYPE(GstPylonObject, gst_pylon_object, GST, PYLON_OBJECT,
                         GstObject)

struct _GstPylonObjectClass {
  GstObjectClass parent_class;
};

typedef struct {
  gint width;
  gint height;
  gint offsetx;
  gint offsety;
} dimension_t;

typedef struct {
  std::shared_ptr<Pylon::CBaslerUniversalInstantCamera> camera;
  GenApi::INodeMap* nodemap;
  gboolean enable_correction;
  dimension_t dimension_cache;
} GstPylonObjectPrivate;

typedef struct {
  const std::string& device_name;
  GstPylonCache& feature_cache;
  GenApi::INodeMap& nodemap;
} GstPylonObjectDeviceMembers;

EXT_PYLONSRC_API GType gst_pylon_object_register(const std::string& device_name,
                                                 GstPylonCache& feature_cache,
                                                 GenApi::INodeMap& nodemap);
EXT_PYLONSRC_API GObject* gst_pylon_object_new(
    std::shared_ptr<Pylon::CBaslerUniversalInstantCamera> camera,
    const std::string& device_name, GenApi::INodeMap* nodemap,
    gboolean enable_correction);

EXT_PYLONSRC_API void gst_pylon_object_set_pylon_selector(
    GenApi::INodeMap& nodemap, const gchar* selector_name,
    gint64& selector_value);

EXT_PYLONSRC_API gpointer
gst_pylon_object_get_instance_private(GstPylonObject* self);

#endif
