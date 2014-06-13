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
#include <limits>

namespace Xapian {

typedef std::map<std::string, double> FeatureVector;

class Document;
class ClusteringAlgorithm;

class XAPIAN_VISIBILITY_DEFAULT ClusterAssignment {
  protected:
    std::map<Xapian::docid, int> doc_to_cluster;
    std::map<int, std::vector<Xapian::docid> > cluster_to_docs;
    friend class ClusteringAlgorithm;
  public:
    int getClusterByDoc(Xapian::docid) const;
    const std::vector<Xapian::docid>& getDocsByCluster(int) const;

};

class XAPIAN_VISIBILITY_DEFAULT FeatureVectorBuilder {
  protected:
    std::map<Xapian::docid, FeatureVector> feature_vectors;
  public:
    virtual void buildFeatureVectors(Xapian::MSet) = 0;
    const FeatureVector& getVectorByDocid(Xapian::docid) const;
};

class XAPIAN_VISIBILITY_DEFAULT TFIDF : public FeatureVectorBuilder{
  protected:
    std::map<std::string, int> termfreqs;
  private:
    void computeTF(Xapian::MSet);
    const FeatureVector& computeTFIDF(Xapian::MSet) const;
  public:
    virtual void buildFeatureVectors(Xapian::MSet);
};

class XAPIAN_VISIBILITY_DEFAULT SimilarityMetric {
  public:
    SimilarityMetric() {
    }
    virtual double similarity(const FeatureVector&, const FeatureVector&) const = 0;
};

class XAPIAN_VISIBILITY_DEFAULT CosineSimilarity : public SimilarityMetric {
  public:
    double similarity(const FeatureVector&, const FeatureVector&) const;
};

class XAPIAN_VISIBILITY_DEFAULT ClusteringAlgorithm {
  protected:
    Xapian::MSet mset;
    ClusterAssignment results;
    void clearClusterToDocs();
    void setClusterForDoc(Xapian::docid, int);
    void addDocForCluster(int, Xapian::docid);
  public:
    ClusteringAlgorithm(Xapian::MSet _mset) : mset(_mset) {
    }
    virtual void cluster() = 0;
    const ClusterAssignment& getResults() const;
};

class XAPIAN_VISIBILITY_DEFAULT KMeans : public ClusteringAlgorithm {
  protected:
    int cluster_count;
    int max_iter;
    std::vector<FeatureVector> centroids;
    SimilarityMetric* metric;
    FeatureVectorBuilder* builder;
  private:
    void init_centroids();
    void assign_centroids();
    void compute_centroids();
  public:
    KMeans(Xapian::MSet _mset, int _cluster_count, int _max_iter,
            SimilarityMetric* _metric, FeatureVectorBuilder* _builder) :
            ClusteringAlgorithm(_mset), cluster_count(_cluster_count),
            max_iter(_max_iter), metric(_metric), builder(_builder) {
    }
    void cluster();
};

}

#endif // XAPIAN_INCLUDED_CLUSTER_H
