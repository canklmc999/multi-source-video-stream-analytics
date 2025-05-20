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

#include "gstpyloncache.h"

#include <errno.h>
#include <glib/gfileutils.h>
#include <gst/pylon/gstpylonincludes.h>

#define DIRERR -1

/* prototypes */
static std::string gst_pylon_cache_create_filepath(
    const std::string &cache_filename);

static std::string gst_pylon_cache_create_filepath(
    const std::string &cache_filename) {
  gchar *filename_hash =
      g_compute_checksum_for_string(G_CHECKSUM_SHA256, cache_filename.c_str(),
                                    strlen(cache_filename.c_str()));
  std::string filename_hash_str = filename_hash;
  g_free(filename_hash);

  std::string dirpath = std::string(g_get_user_cache_dir()) + "/" + "gstpylon";

  /* Create gstpylon directory */
  gint dir_permissions = 0775;
  gint ret = g_mkdir_with_parents(dirpath.c_str(), dir_permissions);
  std::string filepath = dirpath + "/" + filename_hash_str + ".config";
  if (DIRERR == ret) {
    std::string msg =
        "Failed to create " + dirpath + ": " + std::string(strerror(errno));
    GST_WARNING("%s", msg.c_str());
  }

  return filepath;
}

GstPylonCache::GstPylonCache(const std::string &name)
    : filepath(gst_pylon_cache_create_filepath(name)),
      feature_cache_dict(g_key_file_new()),
      is_modified(FALSE) {
  /* load initial cache file */
  if (!LoadCacheFile()) {
    GST_LOG("No feature cache file found");
  }
}

GstPylonCache::~GstPylonCache() { g_key_file_free(this->feature_cache_dict); }

gboolean GstPylonCache::LoadCacheFile() {
  gboolean ret = TRUE;

  /* Check if file exists */
  ret = g_file_test(this->filepath.c_str(), G_FILE_TEST_EXISTS);
  if (!ret) {
    return FALSE;
  }

  /* Check if file contents are valid, this also sets the content of the
   * GKeyFile object if valid */
  ret = g_key_file_load_from_file(
      this->feature_cache_dict, this->filepath.c_str(), G_KEY_FILE_NONE, NULL);
  if (!ret) {
    return FALSE;
  }
  return TRUE;
}

gboolean GstPylonCache::HasNewSettings() { return is_modified; }

void GstPylonCache::CreateCacheFile() {
  GError *file_err = NULL;

#if defined(GLIB_VERSION_2_66) && GLIB_VERSION_MIN_REQUIRED >= GLIB_VERSION_2_66
  gchar *contents = NULL;
  gsize length = 0;

  contents = g_key_file_to_data(this->feature_cache_dict, &length, NULL);
  g_assert(contents != NULL);

  gboolean ret = g_file_set_contents_full(
      this->filepath.c_str(), contents, length,
      static_cast<GFileSetContentsFlags>(G_FILE_SET_CONTENTS_CONSISTENT), 0666,
      &file_err);

  g_free(contents);
#else
  gboolean ret = g_key_file_save_to_file(this->feature_cache_dict,
                                         this->filepath.c_str(), &file_err);
#endif

  if (!ret) {
    std::string file_err_str = file_err->message;
    g_error_free(file_err);
    throw Pylon::GenericException(file_err_str.c_str(), __FILE__, __LINE__);
  }
}

void GstPylonCache::SetIntegerAttribute(const char *feature,
                                        const char *attribute,
                                        const gint64 val) {
  g_key_file_set_int64(this->feature_cache_dict, feature, attribute, val);
  is_modified = true;
}

void GstPylonCache::SetDoubleAttribute(const char *feature,
                                       const char *attribute,
                                       const gdouble val) {
  g_key_file_set_double(this->feature_cache_dict, feature, attribute, val);
  is_modified = true;
}

bool GstPylonCache::GetIntegerAttribute(const char *feature,
                                        const char *attribute, gint64 &val) {
  GError *err = NULL;

  gint64 value =
      g_key_file_get_int64(this->feature_cache_dict, feature, attribute, &err);
  if (err) {
    GST_WARNING("Could not read values for feature %s from file %s: %s",
                feature, this->filepath.c_str(), err->message);
    g_error_free(err);
    return false;
  }
  val = value;
  return true;
}

bool GstPylonCache::GetDoubleAttribute(const char *feature,
                                       const char *attribute, gdouble &val) {
  GError *err = NULL;

  gdouble value =
      g_key_file_get_double(this->feature_cache_dict, feature, attribute, &err);
  if (err) {
    GST_WARNING("Could not read values for feature %s from file %s: %s",
                feature, this->filepath.c_str(), err->message);
    g_error_free(err);
    return false;
  }
  val = value;
  return true;
}

void GstPylonCache::SetIntProps(const gchar *feature_name, const gint64 min,
                                const gint64 max, const GParamFlags flags) {
  SetIntegerAttribute(feature_name, "min", min);
  SetIntegerAttribute(feature_name, "max", max);
  SetIntegerAttribute(feature_name, "flags", static_cast<gint64>(flags));
}
void GstPylonCache::SetDoubleProps(const gchar *feature_name, const gdouble min,
                                   const gdouble max, const GParamFlags flags) {
  SetDoubleAttribute(feature_name, "min", min);
  SetDoubleAttribute(feature_name, "max", max);
  SetIntegerAttribute(feature_name, "flags", static_cast<gint64>(flags));
}

bool GstPylonCache::GetIntProps(const gchar *feature_name, gint64 &min,
                                gint64 &max, GParamFlags &flags) {
  if (!GetIntegerAttribute(feature_name, "min", min)) return false;
  if (!GetIntegerAttribute(feature_name, "max", max)) return false;
  gint64 flag_val = 0;
  if (!GetIntegerAttribute(feature_name, "flags", flag_val)) return false;

  flags = static_cast<GParamFlags>(flag_val);

  return true;
}

bool GstPylonCache::GetDoubleProps(const char *feature_name, gdouble &min,
                                   gdouble &max, GParamFlags &flags) {
  if (!GetDoubleAttribute(feature_name, "min", min)) return false;
  if (!GetDoubleAttribute(feature_name, "max", max)) return false;

  gint64 flag_val = 0;
  if (!GetIntegerAttribute(feature_name, "flags", flag_val)) return false;

  flags = static_cast<GParamFlags>(flag_val);

  return true;
}
