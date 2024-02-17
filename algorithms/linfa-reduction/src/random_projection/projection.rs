/// Macro that implements [`linfa::traits::Transformer`]
/// for [`GaussianRandomProjection`] and [`SparseRandomProjection`],
/// to avoid some code duplication.
#[macro_export]
macro_rules! impl_proj {
    ($t:ty) => {
        impl<F, D> ::linfa::traits::Transformer<&::ndarray::ArrayBase<D, Ix2>, ::ndarray::Array2<F>>
            for $t
        where
            F: ::linfa::Float,
            D: ::ndarray::Data<Elem = F>,
        {
            /// Compute the embedding of a two-dimensional array
            fn transform(&self, x: &::ndarray::ArrayBase<D, Ix2>) -> ::ndarray::Array2<F> {
                x.dot(&self.projection)
            }
        }

        impl<F, D> ::linfa::traits::Transformer<::ndarray::ArrayBase<D, Ix2>, ::ndarray::Array2<F>>
            for $t
        where
            F: ::linfa::Float,
            D: ::ndarray::Data<Elem = F>,
        {
            /// Compute the embedding of a two-dimensional array
            fn transform(&self, x: ::ndarray::ArrayBase<D, Ix2>) -> ::ndarray::Array2<F> {
                self.transform(&x)
            }
        }

        impl<F: Float, T: ::linfa::prelude::AsTargets>
            ::linfa::traits::Transformer<
                ::linfa::DatasetBase<::ndarray::Array2<F>, T>,
                ::linfa::DatasetBase<::ndarray::Array2<F>, T>,
            > for $t
        {
            /// Compute the embedding of a dataset
            ///
            /// # Parameter
            ///
            /// * `data`: a dataset
            ///
            /// # Returns
            ///
            /// New dataset, with data equal to the embedding of the input data
            fn transform(
                &self,
                data: ::linfa::DatasetBase<::ndarray::Array2<F>, T>,
            ) -> ::linfa::DatasetBase<::ndarray::Array2<F>, T> {
                let new_records = self.transform(data.records().view());

                ::linfa::DatasetBase::new(new_records, data.targets)
            }
        }

        impl<
                'a,
                F: ::linfa::Float,
                L: 'a,
                T: ::linfa::prelude::AsTargets<Elem = L> + ::linfa::dataset::FromTargetArray<'a>,
            >
            ::linfa::traits::Transformer<
                &'a ::linfa::DatasetBase<::ndarray::Array2<F>, T>,
                ::linfa::DatasetBase<::ndarray::Array2<F>, T::View>,
            > for $t
        {
            /// Compute the embedding of a dataset
            ///
            /// # Parameter
            ///
            /// * `data`: a dataset
            ///
            /// # Returns
            ///
            /// New dataset, with data equal to the embedding of the input data
            fn transform(
                &self,
                data: &'a ::linfa::DatasetBase<::ndarray::Array2<F>, T>,
            ) -> ::linfa::DatasetBase<::ndarray::Array2<F>, T::View> {
                let new_records = self.transform(data.records().view());

                ::linfa::DatasetBase::new(
                    new_records,
                    T::new_targets_view(::linfa::prelude::AsTargets::as_targets(data)),
                )
            }
        }
    };
}
