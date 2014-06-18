/** \file clustering.h
 * \brief API for clustering groups of documents.
 */
/* Copyright 2014 George Daniel MITRA
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301 USA
 */

#ifndef XAPIAN_INCLUDED_CLUSTER_H
#define XAPIAN_INCLUDED_CLUSTER_H

#include <xapian/database.h>
#include <xapian/enquire.h>
#include <xapian/error.h>
#include <xapian/visibility.h>

#include <map>
#include <string>
#include <vector>

namespace Xapian {

/** The feature vector for a document
 *
 *  The feature vector is an n-dimensional vector which contains a numerical
 *  feature for each term in the document
 */
typedef std::map<std::string, double> feature_vector;

class Document;
class ClusteringAlgorithm;

class XAPIAN_VISIBILITY_DEFAULT ClusterAssignment
{
  protected:
    std::map<docid, clusterid> doc_to_cluster;
    std::map<clusterid, std::vector<docid> > cluster_to_docs;
    friend class ClusteringAlgorithm;
  public:
    clusterid get_cluster_by_doc(docid) const;
    const std::vector<docid>& get_docs_by_cluster(clusterid) const;
};

class XAPIAN_VISIBILITY_DEFAULT FeatureVectorBuilder
{
  protected:
    std::map<docid, feature_vector> feature_vectors;
  public:
    virtual void build_feature_vectors(MSet) = 0;
    const feature_vector& get_vector_by_docid(docid) const;
};

class XAPIAN_VISIBILITY_DEFAULT TFIDF : public FeatureVectorBuilder
{
  protected:
    std::map<std::string, termcount> termfreqs;
  private:
    void compute_tf(MSet);
    const feature_vector& compute_tfidf(MSet) const;
  public:
    virtual void build_feature_vectors(MSet);
};

class XAPIAN_VISIBILITY_DEFAULT SimilarityMetric {
  public:
    SimilarityMetric() {
    }
    virtual double similarity(const feature_vector&, const feature_vector&) const = 0;
};

class XAPIAN_VISIBILITY_DEFAULT CosineSimilarity : public SimilarityMetric {
  public:
    double similarity(const feature_vector&, const feature_vector&) const;
};

class XAPIAN_VISIBILITY_DEFAULT ClusteringAlgorithm {
  protected:
    MSet mset;
    ClusterAssignment results;
    void clear_cluster_to_docs();
    void set_cluster_for_doc(docid, clusterid);
    void add_doc_for_cluster(clusterid, docid);
  public:
    ClusteringAlgorithm(MSet _mset) : mset(_mset) {
    }
    virtual void cluster() = 0;
    const ClusterAssignment& get_results() const;
};

class XAPIAN_VISIBILITY_DEFAULT KMeans : public ClusteringAlgorithm {
  protected:
    clustercount cluster_count;
    unsigned int max_iter;
    std::vector<feature_vector> centroids;
    SimilarityMetric* metric;
    FeatureVectorBuilder* builder;
  private:
    void init_centroids();
    void assign_centroids();
    void compute_centroids();
  public:
    KMeans(MSet _mset, clustercount _cluster_count, unsigned int _max_iter,
            SimilarityMetric* _metric, FeatureVectorBuilder* _builder) :
            ClusteringAlgorithm(_mset), cluster_count(_cluster_count),
            max_iter(_max_iter), metric(_metric), builder(_builder) {
    }
    void cluster();
};

}

#endif // XAPIAN_INCLUDED_CLUSTER_H
